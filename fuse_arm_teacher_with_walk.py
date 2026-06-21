#!/usr/bin/env python3
"""Fuse a full-H1 walking checkpoint with an upper-arm swing teacher checkpoint.

The fusion is intentionally conservative: it keeps the walking checkpoint as the
base payload, then replaces only selected action-head rows and action std entries
for the arm DOFs. Hidden layers, critic, and lower-body action rows stay from the
walking checkpoint, so the fused checkpoint should normally be fine-tuned with
``algo.config.load_optimizer=False`` before final evaluation.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path
from typing import Any

import torch


H1_DOF_NAMES = [
    "left_hip_yaw_joint",
    "left_hip_roll_joint",
    "left_hip_pitch_joint",
    "left_knee_joint",
    "left_ankle_joint",
    "right_hip_yaw_joint",
    "right_hip_roll_joint",
    "right_hip_pitch_joint",
    "right_knee_joint",
    "right_ankle_joint",
    "torso_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
]

ARM_SWING_DOF_NAMES = [
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a fused full-H1 checkpoint by copying upper-arm action-head "
            "rows from an arm-swing teacher into a walking checkpoint."
        )
    )
    parser.add_argument("--walk-checkpoint", required=True, help="Full-H1 walking checkpoint path.")
    parser.add_argument(
        "--arm-checkpoint",
        "--arm-teacher-checkpoint",
        dest="arm_checkpoint",
        required=True,
        help="Full-H1 arm-swing teacher checkpoint path.",
    )
    parser.add_argument("--output", required=True, help="Output fused checkpoint path.")
    parser.add_argument(
        "--upper-dofs",
        nargs="+",
        default=ARM_SWING_DOF_NAMES,
        help="Action DOF rows to copy from the teacher. Defaults to shoulder pitch only; pass --upper-dofs explicitly to include elbows.",
    )
    parser.add_argument(
        "--include-torso",
        action="store_true",
        help="Also copy the torso action row from the teacher.",
    )
    parser.add_argument(
        "--blend",
        type=float,
        default=1.0,
        help="Teacher blend for selected rows: 1.0 copies teacher rows, 0.5 averages with walking rows.",
    )
    parser.add_argument(
        "--no-copy-config",
        action="store_true",
        help="Do not copy the walking run config.yaml beside the fused checkpoint.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and print planned fusion without writing.")
    return parser.parse_args()


def torch_load(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"{path} is not a torch checkpoint dict")
    return payload


def require_actor(payload: dict[str, Any], path: Path) -> dict[str, torch.Tensor]:
    actor = payload.get("actor_model_state_dict")
    if not isinstance(actor, dict):
        raise KeyError(f"{path} has no actor_model_state_dict")
    return actor


def find_std_key(actor: dict[str, torch.Tensor]) -> str:
    if "std" in actor and getattr(actor["std"], "ndim", None) == 1:
        return "std"
    candidates = [key for key, value in actor.items() if getattr(value, "ndim", None) == 1 and key.endswith("std")]
    if len(candidates) != 1:
        raise ValueError(f"Could not uniquely identify actor std key; candidates={candidates}")
    return candidates[0]


def find_action_head(actor: dict[str, torch.Tensor], action_dim: int) -> tuple[str, str]:
    candidates = []
    for key, value in actor.items():
        if not key.endswith("weight") or getattr(value, "ndim", None) != 2:
            continue
        if value.shape[0] != action_dim:
            continue
        bias_key = key[: -len("weight")] + "bias"
        bias = actor.get(bias_key)
        if getattr(bias, "shape", None) == torch.Size([action_dim]):
            candidates.append((key, bias_key))
    if len(candidates) != 1:
        raise ValueError(f"Could not uniquely identify action head; candidates={candidates}")
    return candidates[0]


def check_same_shape(name: str, walk_actor: dict[str, torch.Tensor], arm_actor: dict[str, torch.Tensor]) -> None:
    if name not in arm_actor:
        raise KeyError(f"Teacher actor is missing key {name}")
    if walk_actor[name].shape != arm_actor[name].shape:
        raise ValueError(
            f"Shape mismatch for {name}: walk={tuple(walk_actor[name].shape)}, "
            f"teacher={tuple(arm_actor[name].shape)}"
        )


def selected_indices(dof_names: list[str], include_torso: bool) -> tuple[list[str], list[int]]:
    selected = list(dof_names)
    if include_torso and "torso_joint" not in selected:
        selected.insert(0, "torso_joint")

    name_to_index = {name: index for index, name in enumerate(H1_DOF_NAMES)}
    unknown = [name for name in selected if name not in name_to_index]
    if unknown:
        raise ValueError(f"Unknown H1 DOF names: {unknown}")
    indices = [name_to_index[name] for name in selected]
    return selected, indices


def blend_rows(base: torch.Tensor, source: torch.Tensor, indices: list[int], blend: float) -> torch.Tensor:
    result = base.clone()
    result[indices] = (1.0 - blend) * base[indices] + blend * source[indices]
    return result


def maybe_copy_config(walk_checkpoint: Path, output: Path) -> str | None:
    source = walk_checkpoint.parent / "config.yaml"
    if not source.exists():
        return None
    target = output.parent / "config.yaml"
    if source.resolve() == target.resolve():
        return str(target)
    if not target.exists():
        shutil.copy2(source, target)
    return str(target)


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.blend <= 1.0:
        raise ValueError("--blend must be between 0.0 and 1.0")

    walk_path = Path(args.walk_checkpoint)
    arm_path = Path(args.arm_checkpoint)
    output_path = Path(args.output)
    if output_path.resolve(strict=False) in {
        walk_path.resolve(strict=False),
        arm_path.resolve(strict=False),
    }:
        raise ValueError("Output path must not overwrite an input checkpoint")

    walk = torch_load(walk_path)
    arm = torch_load(arm_path)
    walk_actor = require_actor(walk, walk_path)
    arm_actor = require_actor(arm, arm_path)

    std_key = find_std_key(walk_actor)
    check_same_shape(std_key, walk_actor, arm_actor)
    action_dim = int(walk_actor[std_key].numel())
    if action_dim != len(H1_DOF_NAMES):
        raise ValueError(
            f"Expected full-H1 action_dim={len(H1_DOF_NAMES)}, got {action_dim}. "
            "Use a full-H1 walking checkpoint, not a 10DoF baseline checkpoint."
        )

    weight_key, bias_key = find_action_head(walk_actor, action_dim)
    check_same_shape(weight_key, walk_actor, arm_actor)
    check_same_shape(bias_key, walk_actor, arm_actor)

    dof_names, indices = selected_indices(args.upper_dofs, args.include_torso)
    fusion_info = {
        "mode": "action_head_row_fusion",
        "walk_checkpoint": str(walk_path),
        "arm_checkpoint": str(arm_path),
        "output": str(output_path),
        "blend": args.blend,
        "action_dim": action_dim,
        "action_head_weight_key": weight_key,
        "action_head_bias_key": bias_key,
        "std_key": std_key,
        "fused_dof_names": dof_names,
        "fused_action_indices": indices,
        "note": "Hidden layers and critic are kept from the walking checkpoint; fine-tune with algo.config.load_optimizer=False.",
    }

    print(json.dumps(fusion_info, indent=2))
    if args.dry_run:
        print("Dry run only; no checkpoint written.")
        return

    fused = copy.deepcopy(walk)
    fused_actor = fused["actor_model_state_dict"]
    fused_actor[weight_key] = blend_rows(walk_actor[weight_key], arm_actor[weight_key], indices, args.blend)
    fused_actor[bias_key] = blend_rows(walk_actor[bias_key], arm_actor[bias_key], indices, args.blend)
    fused_actor[std_key] = blend_rows(walk_actor[std_key], arm_actor[std_key], indices, args.blend)
    fused["fusion_info"] = fusion_info
    if isinstance(fused.get("infos"), dict):
        fused["infos"]["fusion_info"] = fusion_info

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.no_copy_config:
        copied_config = maybe_copy_config(walk_path, output_path)
        fusion_info["copied_config"] = copied_config
        if copied_config is None:
            print("Warning: walking checkpoint config.yaml was not found; eval may need a config beside the fused checkpoint.")

    torch.save(fused, output_path)
    print(f"Saved fused checkpoint to {output_path}")
    print("Use it for fine-tuning with checkpoint=<fused.pt> algo.config.load_optimizer=False")


if __name__ == "__main__":
    main()
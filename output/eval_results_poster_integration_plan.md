# Eval Results Poster Integration Plan

## Recommended Role on the Poster

The new evaluation results should become the main quantitative evidence panel of the poster. The panel should be titled:

**Quantitative Robustness under Upper-Body Action Pulses**

This panel should sit close to the temporal montage row that shows perturbation recovery. The visual logic is: Figure 1 shows that the behavior looks stable; the quantitative panel proves that the stability holds across 32 parallel environments for a 20 s finite horizon.

The full 11-column table from `eval_results_for_paper_agent.md` is too wide for the poster. The poster should use one compact table plus one short metric callout, not the full paper-style table.

## Main Table for Poster

Use this reduced table:

| Policy / setting | Amp | Survival ↑ | Mean survival ↑ | Falls ↓ | Tracking err. ↓ | Max tilt ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Robust policy | 0.0 | 100% | 20.00 s | 0 | 0.0228 m/s | 4.67° |
| Robust policy | 1.0 | 100% | 20.00 s | 0 | 0.0225 m/s | 7.81° |
| Robust policy | 2.0 | 100% | 20.00 s | 0 | 0.0245 m/s | 9.12° |
| Arm-swing baseline | 2.0 | 0% | 5.48 s | 95 | 0.1401 m/s | 89.93° |

Use `Robust policy` and `Arm-swing baseline` as reader-facing names. If internal names are needed, place them in small gray text as `Stage8` and `Stage7 baseline`, but they should not be the visual headline.

## Metric Callout

Place a large numeric callout beside or above the table:

**100% survival at amp = 2.0**

Small subline:

32 environments · 20 s horizon · 0.6 m/s command · torso/shoulder/elbow action pulses

The second callout can be:

**0 falls vs. 95 falls**

Subline:

Robust policy vs. arm-swing baseline under the same amp2 perturbation protocol.

## Optional Mini-Trend

If there is room, add a tiny three-point trend for Stage8 only:

- Tracking error: `0.0228 → 0.0225 → 0.0245 m/s`
- Max tilt: `4.67° → 7.81° → 9.12°`

This should be drawn as two small sparklines or a compact two-row metric strip, not as a large chart. The important message is that stronger perturbations increase tilt, but not falls.

## Arm-Swing Retention

Do not make arm-swing metrics the main quantitative result, because the pre-robust baseline has larger endpoint sagittal motion but fails under perturbation. Use them as a small supporting sentence:

At amp2, the robust policy still retains visible upper-body motion, with shoulder-pitch swing amplitude of `0.2536 rad` and elbow endpoint sagittal separation of `0.1165 m`.

This supports the claim that robustness is not obtained merely by freezing the upper body.

## What Not to Emphasize

Do not headline recovery success rate, because the Stage7 baseline has a high recovery-success number despite repeated resets and complete failure over the full horizon. Survival, mean survival time, total falls, tracking error, and max tilt are more reliable for the baseline comparison.

Do not quantitatively compare against the 10DoF lower-body baseline for upper-body robustness, because it does not expose the same 19DoF upper-body action space.

Do not claim sim-to-real robustness or long-horizon benchmark performance. This is a finite-horizon simulation evaluation for the course poster.

## Suggested Caption

Quantitative perturbation evaluation over 32 parallel H1 humanoids for a 20 s horizon at a commanded forward velocity of 0.6 m/s. Upper-body action pulses are injected into torso, shoulder, and elbow joints. The robust policy preserves full survival and low tracking error up to amp2, while the pre-robust arm-swing baseline fails under the same perturbation.


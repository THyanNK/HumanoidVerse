import { spawn } from "node:child_process";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const chromePath = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const port = 9251;
const htmlPath = path.join(__dirname, "method_overview.html");
const pdfPath = path.join(__dirname, "method_overview.pdf");
const pngPath = path.join(__dirname, "method_overview.png");
const profileDir = await fs.mkdtemp(path.join(os.tmpdir(), "method-overview-chrome-"));
const fileUrl = `file://${htmlPath}`;

const chrome = spawn(chromePath, [
  "--headless=new",
  "--disable-gpu",
  "--no-first-run",
  "--no-default-browser-check",
  `--remote-debugging-port=${port}`,
  `--user-data-dir=${profileDir}`,
  "about:blank",
], { stdio: ["ignore", "ignore", "pipe"] });

let stderr = "";
chrome.stderr.on("data", chunk => { stderr += chunk.toString(); });

const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

async function getJson(url, init) {
  const response = await fetch(url, init);
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}: ${await response.text()}`);
  }
  return response.json();
}

async function waitForChrome() {
  const deadline = Date.now() + 15000;
  while (Date.now() < deadline) {
    try {
      return await getJson(`http://127.0.0.1:${port}/json/version`);
    } catch {
      await sleep(200);
    }
  }
  throw new Error(`Chrome did not expose CDP on port ${port}.\n${stderr}`);
}

function connect(wsUrl) {
  const ws = new WebSocket(wsUrl);
  let nextId = 1;
  const pending = new Map();
  const eventWaiters = new Map();

  ws.addEventListener("message", event => {
    const message = JSON.parse(event.data);
    if (message.id && pending.has(message.id)) {
      const { resolve, reject } = pending.get(message.id);
      pending.delete(message.id);
      message.error ? reject(new Error(JSON.stringify(message.error))) : resolve(message.result);
      return;
    }
    if (message.method && eventWaiters.has(message.method)) {
      for (const resolve of eventWaiters.get(message.method)) resolve(message.params ?? {});
      eventWaiters.delete(message.method);
    }
  });

  const opened = new Promise((resolve, reject) => {
    ws.addEventListener("open", resolve, { once: true });
    ws.addEventListener("error", reject, { once: true });
  });

  return {
    opened,
    send(method, params = {}) {
      const id = nextId++;
      ws.send(JSON.stringify({ id, method, params }));
      return new Promise((resolve, reject) => pending.set(id, { resolve, reject }));
    },
    waitFor(method) {
      return new Promise(resolve => {
        const waiters = eventWaiters.get(method) ?? [];
        waiters.push(resolve);
        eventWaiters.set(method, waiters);
      });
    },
    close() { ws.close(); },
  };
}

try {
  await waitForChrome();
  const target = await getJson(
    `http://127.0.0.1:${port}/json/new?${encodeURIComponent(fileUrl)}`,
    { method: "PUT" },
  );
  const cdp = connect(target.webSocketDebuggerUrl);
  await cdp.opened;
  await cdp.send("Page.enable");
  await cdp.send("Emulation.setEmulatedMedia", { media: "print" });
  await cdp.send("Emulation.setDeviceMetricsOverride", {
    width: 3400,
    height: 2000,
    deviceScaleFactor: 1,
    mobile: false,
  });

  const load = cdp.waitFor("Page.loadEventFired");
  await cdp.send("Page.navigate", { url: fileUrl });
  await load;
  await sleep(600);

  const pdf = await cdp.send("Page.printToPDF", {
    landscape: true,
    displayHeaderFooter: false,
    printBackground: true,
    preferCSSPageSize: true,
    paperWidth: 17,
    paperHeight: 10,
    marginTop: 0,
    marginRight: 0,
    marginBottom: 0,
    marginLeft: 0,
    scale: 1,
  });
  await fs.writeFile(pdfPath, Buffer.from(pdf.data, "base64"));

  const screenshot = await cdp.send("Page.captureScreenshot", {
    format: "png",
    fromSurface: true,
    captureBeyondViewport: false,
    clip: { x: 0, y: 0, width: 1700, height: 1000, scale: 2 },
  });
  await fs.writeFile(pngPath, Buffer.from(screenshot.data, "base64"));

  cdp.close();
  console.log(`Wrote ${pdfPath}`);
  console.log(`Wrote ${pngPath}`);
} finally {
  chrome.kill();
  await sleep(300);
  for (let attempt = 0; attempt < 4; attempt += 1) {
    try {
      await fs.rm(profileDir, { recursive: true, force: true });
      break;
    } catch (error) {
      if (attempt === 3) throw error;
      await sleep(250);
    }
  }
}

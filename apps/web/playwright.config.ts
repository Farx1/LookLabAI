import { defineConfig, devices } from "@playwright/test";
import fs from "node:fs";

// Use a pre-installed Chromium when present (CI images, sandboxes); otherwise Playwright's own.
const localChromium = ["/opt/pw-browsers/chromium/chrome-linux/chrome", "/opt/pw-browsers/chromium"].find((p) => {
  try { return fs.statSync(p).isFile(); } catch { return false; }
});

export default defineConfig({
  testDir: "./e2e",
  timeout: 60_000,
  retries: 0,
  use: {
    baseURL: process.env.E2E_BASE_URL ?? "http://127.0.0.1:3000",
    launchOptions: localChromium ? { executablePath: localChromium } : {},
    screenshot: "only-on-failure",
  },
  projects: [
    { name: "desktop", use: { ...devices["Desktop Chrome"], viewport: { width: 1360, height: 900 } } },
    { name: "mobile", use: { ...devices["Pixel 7"] } },
  ],
});

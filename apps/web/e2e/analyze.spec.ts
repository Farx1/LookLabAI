import { expect, test } from "@playwright/test";

// The analysis job is stubbed (no GPU, no Ollama): the page, consent gate, selection, save, try-on and variations are real.
const PNG = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAACAAAAAwCAIAAAD/zu84AAAAPklEQVR4nO3NMQHAMAzAsK4wwx/IILhPPgmA/c3M2XRX68fggUEySAbJIBkkg2SQDJJBMkgGySAZJIO0PvgBqP4BblOtVfsAAAAASUVORK5CYII=",
  "base64",
);

test("analyze an outfit: consent, pieces, pick a match, save, try on, variations", async ({ page, request }) => {
  const garments = (await (await request.get("/api/garments")).json()) as { id: string; category: string }[];
  const top = garments.find((g) => g.category === "top")!;
  const bottom = garments.find((g) => g.category === "bottom")!;
  const piece = (index: number, g: typeof top) => ({
    index, name: `piece ${index}`, category: g.category, subcategory: null, colors: ["grey"], color_hex: ["#808080"], pattern: "solid",
    bbox: [100 * index, 100, 100 * index + 200, 600], matches: [{ score: 0.9, garment: { ...g, name: `match ${index}`, thumb_url: "", brand: null, price: null, currency: "EUR", product_url: null, is_owned: true, colors: ["grey"] } }],
  });
  const job = { id: "stub", kind: "outfit_analysis", status: "succeeded", provider: null, error: null, result: { pieces: [piece(1, top), piece(2, bottom)], timings: { detect: 1 } } };
  await page.route("**/api/analyze", (r) => (r.request().method() === "POST" ? r.fulfill({ json: { ...job, status: "queued", result: {} } }) : r.continue()));
  await page.route("**/api/jobs/stub", (r) => r.fulfill({ json: job }));

  await page.goto("/analyze");
  await page.waitForLoadState("networkidle");
  await expect(page.getByRole("heading", { name: "What is in this outfit?" })).toBeVisible();
  await expect(page.getByText("Choose an outfit photo")).toHaveCount(0); // consent first
  await page.getByLabel(/This is a photo of me/).check();
  await page.locator('input[type=file]:not([capture])').setInputFiles({ name: "look.png", mimeType: "image/png", buffer: PNG });

  await expect(page.getByRole("group", { name: /piece 1/ })).toBeVisible();
  await expect(page.getByText("2 piece(s)")).toBeVisible();
  await page.getByRole("button", { name: "Save outfit" }).click();
  await expect(page.getByRole("button", { name: "Outfit saved" })).toBeVisible();
  await expect(page.getByRole("link", { name: /Try on/ })).toHaveAttribute("href", new RegExp(`garments=${top.id},${bottom.id}`));
  await page.getByRole("button", { name: "Variations" }).click();
  await expect(page.getByRole("region", { name: "Variations" })).toBeVisible();
});

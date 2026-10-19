import { expect, test } from "@playwright/test";

// Requires a running stack seeded with scripts/seed_demo.py --with-person.
const SHOTS = process.env.E2E_SCREENSHOTS;

async function shot(page: import("@playwright/test").Page, name: string, project: string) {
  if (SHOTS) await page.screenshot({ path: `${SHOTS}/${project}-${name}.png`, fullPage: false });
}

test("principal use case: wardrobe → 15 outfits → save → stylist follow-up → mock try-on", async ({ page }, info) => {
  const p = info.project.name;
  await page.goto("/");
  await expect(page.getByRole("heading", { name: /Dress for the moment/ })).toBeVisible();
  await shot(page, "dashboard", p);

  await page.goto("/wardrobe");
  await expect(page.getByText("cream relaxed blouse")).toBeVisible();
  await shot(page, "wardrobe", p);
  await page.getByRole("button", { name: "Open cream relaxed blouse" }).click();
  await expect(page.getByRole("dialog")).toContainText("Your edits always win");
  await shot(page, "garment-editor", p);
  await page.keyboard.press("Escape");

  await page.goto("/outfits");
  await page.getByPlaceholder(/8 outfits for a summer wedding/).fill(
    "Find 15 outfits for a romantic anniversary dinner. I prefer loose-fitting, comfortable clothes, neutral colors, Korean-inspired fashion and a budget of €100.",
  );
  await page.getByRole("button", { name: "Generate looks" }).click();
  await expect(page.getByText("budget 100 EUR")).toBeVisible();
  await expect(page.locator("article")).toHaveCount(15);
  await expect(page.locator("article img").first()).toHaveJSProperty("complete", true);
  await page.waitForTimeout(800);
  await shot(page, "outfits", p);
  await page.locator("article").first().getByRole("button", { name: "Save" }).click();
  await expect(page.locator("article").first().getByRole("button", { name: "Saved" })).toBeVisible();

  await page.goto("/stylist");
  await page.getByRole("button", { name: "New conversation" }).filter({ visible: true }).first().click();
  const box = page.getByPlaceholder(/Keep the same shoes/);
  await box.fill("5 romantic outfits for a date, neutral colors, budget €100");
  await box.press("Enter");
  await expect(page.getByText(/Here are 5 looks/)).toBeVisible();
  await box.fill("Make them less fitted.");
  await box.press("Enter");
  await expect(page.getByText(/looser fit/).last()).toBeVisible();
  await expect(page.getByText("Remembering")).toBeVisible();
  await shot(page, "stylist", p);

  await page.goto("/studio");
  await expect(page.getByText(/labelled mock preview/)).toBeVisible();
  await page.getByTitle("cream relaxed blouse").click();
  await page.getByRole("button", { name: /Run mock preview/ }).click();
  await expect(page.getByRole("slider", { name: /Compare/ })).toBeVisible({ timeout: 20_000 });
  await expect(page.getByText("Mock — no model run")).toBeVisible();
  await shot(page, "studio", p);

  await page.goto("/saved");
  await expect(page.locator("article").first()).toBeVisible();
  await shot(page, "saved", p);
  await page.goto("/lookbook");
  await expect(page.getByRole("heading", { name: "Date" })).toBeVisible();
  await shot(page, "lookbook", p);
  await page.goto("/settings");
  await page.getByRole("tab", { name: "AI engines" }).click();
  await expect(page.getByText("Virtual try-on engines")).toBeVisible();
  await shot(page, "settings-engines", p);
  await page.getByRole("tab", { name: "Costs & performance" }).click();
  await expect(page.getByText("By engine (last 30 days)")).toBeVisible();
  await shot(page, "settings-costs", p);
});

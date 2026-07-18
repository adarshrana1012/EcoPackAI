import { test, expect } from '@playwright/test';

test.describe('Analytics Page View', () => {
  test('analytics aggregates layout structures', async ({ page }) => {
    await page.goto('/login');
    await page.fill('input[type="email"]', 'demo@ecopackai.io');
    await page.fill('input[type="password"]', 'demo123');
    await page.click('button[type="submit"]');
    
    await page.goto('/analytics');
    const header = page.locator('h1:has-text("Analytics")');
    await expect(header).toBeVisible();
  });
});

const { cfg } = require("../image-extractor/config");

test("configuration test", () => {
  expect(cfg.chatOpenAI).toBeTruthy();
  expect(cfg.geminiAPIKey).toBeTruthy();
  expect(cfg.googleAI).toBeTruthy();
  expect(cfg.googleModel).toBeTruthy();
});

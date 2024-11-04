const {
  convertBase64,
  createTextExtractChain,
  executeStructuredPrompt,
  createBatches,
} = require("../image-extractor/service/textExtraction");
const { cfg } = require("../image-extractor/config");
const fs = require("fs");
const path = require("node:path");

test("base64 test", () => {
  const fileName = "./text.txt";
  fs.writeFileSync(fileName, "This is a lovely string.");
  expect(convertBase64(fileName)).toBe("VGhpcyBpcyBhIGxvdmVseSBzdHJpbmcu");
  fs.unlinkSync(fileName);
});

test("create extract chain test", () => {
  expect(createTextExtractChain(cfg.googleAI)).toBeTruthy();
});

test("execute structured prompt test", async () => {
  const imagePath = path.join(
    __filename,
    "..",
    "..",
    "images",
    "2024-08-22-gift-for-the-soul.jfif",
  );
  console.info("imagePath", imagePath);
  expect(fs.existsSync(imagePath)).toBeTruthy();
  const extracted = await executeStructuredPrompt(cfg.googleAI, imagePath);
  console.info("extracted text", extracted);
  expect(extracted).toBeTruthy();
  expect(extracted["main_text"]).toBeTruthy();
});

test("createBatches test", () => {
  const a = [1, 2, 3, 4, 5, 6, 7];
  const batches = createBatches(a, 2);
  expect(batches.length).toBe(4);
  expect(batches[0].length).toBe(2);
  expect(batches[0]).toStrictEqual([1, 2]);
  expect(batches[1].length).toBe(2);
  expect(batches[1]).toStrictEqual([3, 4]);
  expect(batches[2].length).toBe(2);
  expect(batches[2]).toStrictEqual([5, 6]);
  expect(batches[3].length).toBe(1);
  expect(batches[3]).toStrictEqual([7]);
});

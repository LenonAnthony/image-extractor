const fs = require("fs");
const { ChatPromptTemplate } = require("@langchain/core/prompts");
const { TextExtract } = require("../model/text-extract");

const PROMPT_INSTRUCTION = "Please extract the text from the provided image.";

const prompt = ChatPromptTemplate.fromMessages([
  ["system", PROMPT_INSTRUCTION],
  [
    "user",
    [
      {
        type: "image_url",
        image_url: { url: "data:image/jpeg;base64,{image_data}" },
      },
    ],
  ],
]);

function convertBase64(imagePath) {
  const bytes = fs.readFileSync(imagePath);
  return Buffer.from(bytes).toString("base64");
}

function createTextExtractChain(chatModel) {
  return prompt.pipe(chatModel.withStructuredOutput(TextExtract));
}

function executeStructuredPrompt(chatModel, imagePath) {
  convertedImg = convertBase64(imagePath);
  chain = createTextExtractChain(chatModel);
  return chain.invoke({ image_data: convertedImg });
}

function createBatches(imagePaths, batchSize) {
  const batches = [];
  for (let i = 0; i < imagePaths.length; i += batchSize) {
    batches.push(imagePaths.slice(i, i + batchSize));
  }
  return batches;
}

async function executeBatchStructuredPrompt(chatModel, imagePaths, batchSize) {
  if (batchSize < 0) {
    batchSize = 1;
  }
  const batches = createBatches(imagePaths, batchSize);
  const chain = createTextExtractChain(chatModel);
  const res = [];
  for (const b of batches) {
    const extracts = await chain.batch(
      b.map((img) => ({ image_data: convertBase64(img) })),
    );
    res.push(...extracts)
  }
  return imagePaths.map((path, index) => {
    const extract = res[index];
    extract["path"] = path;
    return extract;
  });
}

module.exports = {
  convertBase64,
  createTextExtractChain,
  executeStructuredPrompt,
  createBatches,
  executeBatchStructuredPrompt,
};

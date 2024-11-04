const { Command } = require("commander");
const program = new Command();
const assert = require("assert");
const fs = require("fs");
const {globSync} = require('glob')
const path = require("node:path");

const { executeStructuredPrompt, executeBatchStructuredPrompt } = require("./service/textExtraction")
const { cfg } = require("./config")

const ALLOWED_MODELS = ["openai", "google"];

const ALLOWED_EXTENSIONS = ["jpg", "jfif", "png"];

const MODEL_MAP = {
    "openai": cfg.chatOpenAI,
    "google": cfg.googleAI
}

const PROGRAMME_NAME = "Image Extractor"

function chooseModel(model) {
    const aiModel = MODEL_MAP[model]
    if(!aiModel) {
        throw new Error(`Cannot find AI model for ${model}.`);
    }
    return aiModel
}

function checkAllowedValues(value, enumeration, paramName) {
  if (!enumeration.includes(value)) {
    throw new Error(
      `Invalid value for --${paramName}. Allowed values are: ${enumeration.join(", ")}`,
    );
  }
  return value;
}

function createTargetFile(image, model) {
    const parent = path.dirname(image)
    const baseName = path.parse(image).name
    return path.join(parent, `${model}_${baseName}.json`)
}

function writeToJson(res, image, model) {
    const resJson = JSON.stringify(res, null, 2);
    const targetFile = createTargetFile(image, model);
    console.info(`Writing to ${targetFile}.`)
    fs.writeFileSync(targetFile, resJson);
}

program
  .name(PROGRAMME_NAME)
  .version("1.0.0")
  .description(
    "Extracts text from images using LLMs, like OpenAI gpt-4o and Gemini Flash.",
  )
  .requiredOption("-f, --folder <path>", "The folder with the images")
  .requiredOption("-m, --model <model>", "The model to use.", (value) => checkAllowedValues(value, ALLOWED_MODELS, "model"))
  .requiredOption("-e, --extension <extension>", "The file extension.", (value) => checkAllowedValues(value, ALLOWED_EXTENSIONS, "extension"))
  .option("-b, --batch_size <number>", "The batch size")
  .action(async (options) => {
    console.time(PROGRAMME_NAME)
    const { folder, model, extension } = options;
    const batchSize = options.batch_size ? parseInt(options.batch_size) : 1
    assert(
      fs.existsSync(folder),
      `Cannot find ${folder}. Please check the location of this folder.`,
    );
    console.info(`Folder:     ${folder}`);
    console.info(`Model:      ${model}`);
    console.info(`Extension:  ${extension}`);
    console.info(`Batch size: ${batchSize}`);
    const aiModel = chooseModel(model)
    const imagesPaths = globSync(`**/*.${extension}`)
    if(batchSize === 1) {
        for(const image of imagesPaths) {
            const res = await executeStructuredPrompt(aiModel, image)
            writeToJson(res, image, model);
        }
    } else {
        results = await executeBatchStructuredPrompt(aiModel, imagesPaths, batchSize)
        for(const res of results) {
            writeToJson(res, res['path'], model);
        }
    }
    console.timeEnd(PROGRAMME_NAME)
  });

program.parse(process.argv);


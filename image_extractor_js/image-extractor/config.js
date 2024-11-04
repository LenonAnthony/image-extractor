require("dotenv").config();
const { ChatOpenAI } = require("@langchain/openai");
const { ChatGoogleGenerativeAI } = require("@langchain/google-genai");
const assert = require("assert");

class Config {
  constructor() {
    this.openAIKey = process.env.OPENAI_API_KEY;
    assert(!!this.openAIKey, "There is no Open AI key");
    console.info("Found Open AI Key.");

    this.openAIModel = process.env.OPENAI_MODEL;
    assert(!!this.openAIModel, "Please specify your OpenAI model");
    console.info(`Using Open AI model ${this.openAIModel}`);

    // Initialize OpenAI client
    this.chatOpenAI = new ChatOpenAI({
      model: this.openAIModel,
      apiKey: this.openAIKey,
      verbose: false
    });

    // Retrieve and assert Google Gemini API key and model
    this.geminiAPIKey = process.env.GEMINI_API_KEY;
    assert(!!this.geminiAPIKey, "Cannot find Gemini API key");
    console.info("Found Gemini AI Key.");

    this.googleModel = process.env.GOOGLE_MODEL;
    assert(!!this.googleModel, "Please specify your Google Gemini model.");
    console.info(`Using Google ${this.googleModel}`);

    // Initialize Google Gemini client
    this.googleAI = new ChatGoogleGenerativeAI({
      model: this.googleModel,
      apiKey: this.geminiAPIKey,
      verbose: false
    });
  }
}

const cfg = new Config();

module.exports = {
  cfg
};

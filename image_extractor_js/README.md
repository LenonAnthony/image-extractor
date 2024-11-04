# Image Extractor

Extracts text from images using LLMs, using OpenAI gpt-4o and Gemini Flash.

You can either convert the images sequentially or in batches using a simple command line tool.

# Install

```
yarn install
```

# Configuration

Please check the attached .env_local file and at least add these two keys to the .env file that should be created in the root folder:

```
OPENAI_API_KEY
GEMINI_API_KEY
```

If you want to use Langsmith add also this key:

```
LANGCHAIN_API_KEY
```

Otherwise please comment out all of the Langsmith keys.

# Testing

```
yarn run test
```

# Formatting code

```
yarn run pretty
```

# Usage examples

```
node .\image-extractor\main.js -f .\images\ -m google -e jfif
node .\image-extractor\main.js --folder .\images\ --model google --extension jfif --batch_size 2
```

```
node .\image-extractor\main.js --folder .\images\ --model openai --extension jfif
node .\image-extractor\main.js --folder .\images\ --model openai --extension jfif --batch_size 2
```

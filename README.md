# Image Extractor

Extracts text from images using LLMs, using OpenAI gpt-4o and Gemini Flash.

This repository contains one Python implementation and another Javascript based implementation.

Both versions support batched conversion and were tested on OpenAI's gpt-4o and Google's gemini-1.5-flash

# Install and use

```
conda create -n image_extractor python=3.13 or python3 -m venv .venv
conda activate image_extractor / source .venv/bin/activate
cd image_extractor
pip install poetry
poetry install
```
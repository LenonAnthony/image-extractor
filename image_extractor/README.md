# Image Extractor

Extracts text from images using LLMs, like OpenAI gpt-4o, Gemini Flash, Claude, Ollama and so on.

You can either convert the images sequentially or in batches using a simple command line tool.

# Install

```
conda create -n image_extractor python=3.13
conda activate image_extractor
cd image_extractor
python dependencies.py
```

# Configuration
- Before start the analysis, make sure you configure the right LLM in `config.py`
- Copy `.env.example` to `.env` and configure your API keys and models
- For Ollama: Make sure Ollama is running locally with a vision-capable model installed

## Ollama Setup
1. Download and install Ollama from https://ollama.ai/
2. Install a vision model: `ollama pull llama3.2-vision`
3. Start Ollama service (usually runs on http://localhost:11434)
4. Set `OLLAMA_BASE_URL` and `OLLAMA_MODEL` in your `.env` file
# Text extraction of images
- To extract text from images, this is the command (change the params if applicable)
```
python extraction_main.py convert-folder --folder database-test --model ollama --extension jpg
```
- To eval the extraction to its original text, these are the commands for essays, lines and words (change the params if applicable)
```
python extraction_eval_essay.py --model ollama --extension jpg --sample-dir database-test --csv-file dataset-name-test.csv

python extraction_eval_lines.py --model ollama --extension jpg --sample-dir database-test --csv-file dataset-name-test.csv

python extraction_eval.py --model ollama --extension jpg --sample-dir database-test --csv-file dataset-name-test.csv
```

# Prediction of essays score
- Example of how to extract essays score from LLM
```
python -m image_extractor.extraction_essay_main evaluate-essays --csv_file test.csv --model ollama --output_dir ./essay_results
```
- Once you have the `essay_results` with the scores in json, run the eval of essays score
```
python -m image_extractor.score_eval_essay \
  --csv test.csv \
  --predictions-dir essay_results \
  --output-dir ./evaluation_results \
  --model anthropic
```


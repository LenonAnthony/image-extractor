# Image Extractor

Extracts text from images using LLMs, like OpenAI gpt-4o, Gemini Flash, Claude and so on.

You can either convert the images sequentially or in batches using a simple command line tool.

# Install

```
conda create -n image_extractor python=3.13 or python3 -m venv .venv
conda activate image_extractor / source .venv/bin/activate
cd image_extractor
pip install poetry
poetry install
```
# Configuration
- Before start the analysis, make sure you configurate the right LLM in `config.py` by 
uncommenting the LLM calls.
- Be aware if you have the exact key names you need to configurate, on your `.env`
# Text extraction of images
- To extract text from images, this is the command (change the params if applicable)
```
python extraction_main.py convert-folder --folder database-test --model vertexai --extension jpg
```
- To eval the extraction to its original text, these are the commands for essays, lines and words (change the params if applicable)
```
python extraction_eval_essay.py --model vertexai --extension jpg --sample-dir database-test --csv-file dataset-name-test.csv

python extraction_eval_lines.py --model openai --extension jpg --sample-dir database-test --csv-file dataset-name-test.csv

python extraction_eval.py --model openai --extension jpg --sample-dir database-test --csv-file dataset-name-test.csv
```

# Prediction of essays score
- Example of how to extract essays score from LLM
```
python -m image_extractor.extraction_essay_main evaluate-essays --csv_file test.csv --model vertexai --output_dir ./essay_results
```
- Once you have the `essay_results` with the scores in json, run the eval of essays score
```
python -m image_extractor.score_eval_essay \
  --csv test.csv \
  --predictions-dir essay_results \
  --output-dir ./evaluation_results \
  --model anthropic
```


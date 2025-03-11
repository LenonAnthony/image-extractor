# Image Extractor

Extracts text from images using LLMs, like OpenAI gpt-4o and Gemini Flash.

You can either convert the images sequentially or in batches using a simple command line tool.

# Install

```
conda create -n image_extractor python=3.13
conda activate image_extractor
pip install -r requirements.txt
```

# Usage examples

```
python .\image_extractor\extraction_main.py convert-folder --folder .\files --model google --extension jfif
python .\image_extractor\extraction_main.py convert-folder --folder .\files --model google --extension jfif --batch_size 2
```

```
python .\image_extractor\extraction_main.py convert-folder --folder .\files --model openai --extension jfif
python .\image_extractor\extraction_main.py convert-folder --folder .\files --model openai --extension jfif --batch_size 2
```

# rodar o eval
```
python extraction_eval.py --model openai --extension jpg --sample-dir dataset-name --csv-file dataset-name-test.csv
```


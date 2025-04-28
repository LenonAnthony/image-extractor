from pathlib import Path
from enum import StrEnum
import click
import time
import json
import os
from image_extractor.service.answer_sheet_extraction import (
    OpenAiAnswerSheetExtractor,
    VertexAiAnswerSheetExtractor,
    GoogleGenAiAnswerSheetExtractor,
    AnthropicAnswerSheetExtractor
)

class Model(StrEnum):
    OPENAI = "openai"
    VERTEXAI = "vertexai"
    GOOGLE_GENAI = "google" 
    ANTHROPIC = "anthropic"

class Extension(StrEnum):
    JPG = "jpg"
    JPEG = "jpeg"
    PNG = "png"
    JFIF = "jfif"

@click.group()
def cli():
    pass

@cli.command()
@click.option(
    "--folder", 
    required=True,
    help="The folder containing the answer sheet images"
)
@click.option(
    "--model",
    default=Model.GOOGLE_GENAI.value,
    type=click.Choice([m.value for m in Model]),
    help="The model to use for answer sheet extraction"
)
@click.option(
    "--extension",
    default=Extension.JPG.value,
    type=click.Choice([e.value for e in Extension]),
    help="The extension of the images to process"
)
@click.option(
    "--batch_size", 
    default=1, 
    help="The size of the batch for processing images"
)
def extract_answer_sheets(folder: str, model: str, extension: str, batch_size: int):
    """Extract answers from answer sheets using AI vision models."""
    start = time.time()
    folder_path = Path(folder)
    
    if not folder_path.exists():
        click.echo(f"Folder {folder} does not exist.")
        return
    
    files = list(folder_path.glob(f"*.{extension}"))
    if not files:
        click.echo(f"No {extension} files found in {folder}.")
        return
    
    click.echo(f"Found {len(files)} {extension} files in {folder}.")
    
    # Create results directory if it doesn't exist
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Create model directory inside results if it doesn't exist
    model_dir = results_dir / model
    model_dir.mkdir(exist_ok=True)
    
    # Determine model type for filenames
    model_type = ""
    if model == Model.OPENAI.value:
        extractor = OpenAiAnswerSheetExtractor()
        model_type = os.getenv("OPENAI_MODEL", "").replace("gpt-", "")
    elif model == Model.VERTEXAI.value:
        extractor = VertexAiAnswerSheetExtractor()
        model_type = os.getenv("GEMINI_MODEL", "").replace("gemini-", "")
    elif model == Model.GOOGLE_GENAI.value:
        extractor = GoogleGenAiAnswerSheetExtractor()
        model_type = os.getenv("GOOGLE_GENAI_MODEL", "").replace("gemini-", "")
    elif model == Model.ANTHROPIC.value:
        extractor = AnthropicAnswerSheetExtractor()
        model_type = os.getenv("ANTHROPIC_MODEL", "").replace("claude-", "")
    else:
        click.echo(f"Unsupported model type: {model}")
        return
    
    # Process files
    for f in files:
        output_file = model_dir / f"{model}_{model_type}_{f.stem}.json"
        
        # Skip if already processed
        if output_file.exists():
            click.echo(f"File {f} was already processed.")
            continue
        
        click.echo(f"Processing file {f}")
        
        start_file = time.time()
        try:
            extract = extractor.extract_answer_sheet(f)
            elapsed_file = time.time() - start_file
            
            # Save result to JSON
            if extract is not None:
                extract_data = extract.model_dump()
                extract_data["elapsed"] = elapsed_file
                json_data = json.dumps(extract_data, indent=2)
                output_file.write_text(json_data, encoding="utf-8")
                click.echo(f"Wrote {output_file}.")
            else:
                click.echo(f"Could not extract answer sheet data from {f}.")
        except Exception as e:
            click.echo(f"Error processing {f}: {e}")
    
    end = time.time()
    click.echo(f"Total elapsed time: {end - start} seconds.")

if __name__ == "__main__":
    cli()

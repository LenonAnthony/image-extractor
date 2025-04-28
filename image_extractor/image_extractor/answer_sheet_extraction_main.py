from pathlib import Path
from enum import StrEnum
import click
import time
import json
import os
import sys

# Adicionar o diretório atual ao caminho do Python para permitir importações absolutas
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


from model.answer_sheet_extract import AnswerSheetExtract
# import extractor classes
from service.answer_sheet_extraction import OpenAiAnswerSheetExtractor, VertexAiAnswerSheetExtractor, AnthropicAnswerSheetExtractor


class Model(StrEnum):
    OPENAI = "openai"
    VERTEXAI = "vertexai"
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
    "--folder", prompt="The folder with the answer sheet images", help="The folder with the answer sheet images"
)
@click.option(
    "--model",
    default=Model.OPENAI.value,
    prompt="The model you want to use",
    type=click.Choice([m.value for m in Model]),
    help="The model type to be used",
)
@click.option(
    "--extension",
    default="jpg",
    prompt="The extension to process",
    type=click.Choice([e.value for e in Extension]),
    help="The extension of the images",
)
@click.option(
    "--batch_size", default=1, help="The size of the batch"
)
def extract_answer_sheets(folder: str, model: str, extension: str, batch_size: int):
    """Extract answers from answer sheet images."""
    start = time.time()
    root = Path(folder)
    assert root.exists(), f"Path {root} does not exist."
    
    extractor = None
    if model == Model.OPENAI.value:
        extractor = OpenAiAnswerSheetExtractor()
        model_type = os.getenv("OPENAI_MODEL", "").replace("gpt-", "")
    elif model == Model.VERTEXAI.value:
        extractor = VertexAiAnswerSheetExtractor()
        model_type = os.getenv("GEMINI_MODEL", "").replace("gemini-", "")
    elif model == Model.ANTHROPIC.value:
        extractor = AnthropicAnswerSheetExtractor()
        model_type = os.getenv("ANTHROPIC_MODEL", "").replace("claude-", "")
    else:
        raise ValueError(f"Unsupported model type: {model}")

    files = list(root.rglob(f"**/*.{extension}"))
    
    # Create output directory if it doesn't exist
    output_dir = root / "results"
    output_dir.mkdir(exist_ok=True)
    
    def check_existing_file(file_path: Path, model: str, model_type: str) -> bool:
        """Check if this file has already been processed."""
        target_file = output_dir / f"{model}_{model_type}_{file_path.stem}.json"
        return target_file.exists()

    for f in files:
        if check_existing_file(f, model, model_type):
            click.echo(f"File {f} was already processed.")
            continue
        
        click.echo(f"Processing file {f}")
        start_file = time.time()
        try:
            extract = extractor.extract_answers(f)
            elapsed_file = time.time() - start_file

            # Add sheet_id if not present
            if extract.sheet_id is None:
                extract.sheet_id = f.stem
            
            # Convert to detection format
            detection_data = extract.to_detection_format()
            
            # Add elapsed time
            detection_data["elapsed"] = elapsed_file
            
            # Save the results
            target_file = output_dir / f"{model}_{model_type}_{f.stem}.json"
            json_data = json.dumps(detection_data, indent=2)
            target_file.write_text(json_data, encoding="utf-8")
            click.echo(f"Wrote {target_file}.")
            
            # Print a summary of results
            click.echo(f"Found {extract.total_questions} questions.")
            for answer in extract.answers:
                click.echo(f"Question {answer.question_number}: {answer.marked_option} (confidence: {answer.confidence:.2f})")
                
        except Exception as e:
            click.echo(f"Error processing {f}: {e}")

    end = time.time()
    click.echo(f"Elapsed time: {end - start} seconds.")

if __name__ == "__main__":
    cli()
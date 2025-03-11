from pathlib import Path
from enum import StrEnum
import click
import time
import json
from image_extractor.service.text_extraction import OpenAiConversion, VertexAiConversation, GoogleVisionConversion
from image_extractor.model.text_extract import TextExtract
import os

class Model(StrEnum):
    OPENAI = "openai"
    VERTEXAI = "vertexai"
    GOOGLE_VISION = "google_vision"

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
    "--folder", prompt="The folder with the images", help="The folder with the images"
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

def convert_folder(folder: str, model: str, extension: str, batch_size: int):
    start = time.time()
    root = Path(folder)
    assert root.exists(), f"Path {root} does not exist."
    
    conversion = None
    if model == Model.OPENAI.value:
        conversion = OpenAiConversion()
    elif model == Model.VERTEXAI.value:
        conversion = VertexAiConversation()
    elif model == Model.GOOGLE_VISION.value:
        conversion = GoogleVisionConversion()
    else:
        raise ValueError(f"Unsupported model type: {model}")

    files = root.rglob(f"**/*.{extension}")

    if model == Model.OPENAI.value:
        model_type = os.getenv("OPENAI_MODEL").replace("gpt-", "")
    elif model == Model.VERTEXAI.value:  
        model_type = os.getenv("GEMINI_MODEL").replace("gemini-", "")
    else:
        model_type = "unknown"

    def check_existing_file(file_path: Path, model: str, model_type: str) -> bool:
        target_file = file_path.parent / f"{model}_{model_type}_{file_path.stem}.json"
        return target_file.exists()

    for f in files:
        if check_existing_file(f, model, model_type):
            click.echo(f"File {f} was already processed.")
            continue

        start_file = time.time()
        extract = conversion.convert_to_text(f)
        elapsed_file = time.time() - start_file

        def process_extract(extract: TextExtract, f: Path, elapsed: float):
            if extract is not None:
                target_file = f.parent / f"{model}_{model_type}_{f.stem}.json"
                extract_data = extract.model_dump()
                extract_data["elapsed"] = elapsed
                json_data = json.dumps(extract_data, indent=2)
                target_file.write_text(json_data, encoding="utf-8")
                click.echo(f"Wrote {target_file}.")
            else:
                click.echo(f"Could not extract text from {f}.")

        process_extract(extract, f, elapsed_file)

    end = time.time()
    click.echo(f"Elapsed time: {end - start} seconds.")

if __name__ == "__main__":
    cli()
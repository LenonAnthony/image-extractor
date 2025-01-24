from pathlib import Path
from enum import StrEnum
import click
import time
import json
from image_extractor.service.text_extraction import OpenAiConversion, VertexAiConversation
from image_extractor.model.text_extract import TextExtract

class Model(StrEnum):
    OPENAI = "openai"
    VERTEXAI = "vertexai"

class Extension(StrEnum):
    JPG = "jpg"
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
    path = Path(folder)
    assert path.exists(), f"Path {path} does not exist."
    conversion = (
        OpenAiConversion() if model == Model.OPENAI.value else VertexAiConversation()
    )
    files = path.rglob(f"**/*.{extension}")

    def process_extract(extract: TextExtract, f: Path, elapsed: float):
        target_file = path / f"{model}_{f.stem}.json"
        extract_data = extract.model_dump()
        extract_data["elapsed"] = elapsed
        json_data = json.dumps(extract_data, indent=2)
        target_file.write_text(json_data, encoding="utf-8")
        click.echo(f"Wrote {target_file}.")

    if batch_size == 1:
        for f in files:
            start_file = time.time()
            extract = conversion.convert_to_text(f)
            elapsed_file = time.time() - start_file
            process_extract(extract, f, elapsed_file)
    elif batch_size > 1:
        click.echo(f"Using batch size {batch_size}.")
        file_extracts = conversion.convert_to_text_batches(list(files), batch_size)
        for file_extract in file_extracts:
            process_extract(file_extract, file_extract.path)
    end = time.time()
    click.echo(f"Elapsed time: {end - start} seconds.")

if __name__ == "__main__":
    cli()

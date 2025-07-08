from pathlib import Path
from enum import StrEnum
import click
import time
<<<<<<< Updated upstream

from image_extractor.service.text_extraction import GoogleAiConversion, OpenAiConversion
=======
import json
from image_extractor.service.text_extraction import OpenAiConversion, VertexAiConversation, GoogleVisionConversion, AnthropicConversion, OllamaConversion
>>>>>>> Stashed changes
from image_extractor.model.text_extract import TextExtract


class Model(StrEnum):
    OPENAI = "openai"
<<<<<<< Updated upstream
    GOOGLE = "google"

=======
    VERTEXAI = "vertexai"
    GOOGLE_VISION = "google_vision"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
>>>>>>> Stashed changes

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
<<<<<<< Updated upstream
    path = Path(folder)
    assert path.exists(), f"Path {path} does not exist."
    conversion = (
        OpenAiConversion() if model == Model.OPENAI.value else GoogleAiConversion()
    )
    files = path.rglob(f"**/*.{extension}")

    def process_extract(extract: TextExtract, f: Path):
        target_file = path / f"{model}_{f.stem}.json"
        target_file.write_text(extract.model_dump_json(indent=2), encoding="utf-8")
        click.echo(f"Wrote {target_file}.")
=======
    root = Path(folder)
    assert root.exists(), f"Path {root} does not exist."
    
    conversion = None
    if model == Model.OPENAI.value:
        conversion = OpenAiConversion()
    elif model == Model.VERTEXAI.value:
        conversion = VertexAiConversation()
    elif model == Model.GOOGLE_VISION.value:
        conversion = GoogleVisionConversion()
    elif model == Model.ANTHROPIC.value:
        conversion = AnthropicConversion()
    elif model == Model.OLLAMA.value:
        conversion = OllamaConversion()
    else:
        raise ValueError(f"Unsupported model type: {model}")

    files = root.rglob(f"**/*.{extension}")

    if model == Model.OPENAI.value:
        model_type = os.getenv("OPENAI_MODEL").replace("gpt-", "")
    elif model == Model.VERTEXAI.value:  
        model_type = os.getenv("GEMINI_MODEL").replace("gemini-", "")
    elif model == Model.ANTHROPIC.value:
        model_type = os.getenv("ANTHROPIC_MODEL").replace("claude-", "")
    elif model == Model.OLLAMA.value:
        model_type = os.getenv("OLLAMA_MODEL", "deepseek-r1:7b").replace(":", "_")
    else:
        model_type = "unknown"

    def check_existing_file(file_path: Path, model: str, model_type: str) -> bool:
        target_file = file_path.parent / f"{model}_{model_type}_{file_path.stem}.json"
        return target_file.exists()

    for f in files:
        if check_existing_file(f, model, model_type):
            click.echo(f"File {f} was already processed.")
            continue
        
        click.echo(f"Processing file {f}")
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
>>>>>>> Stashed changes

    if batch_size == 1:
        for f in files:
            extract = conversion.convert_to_text(f)
            process_extract(extract, f)
    elif batch_size > 1:
        click.echo(f"Using batch size {batch_size}.")
        file_extracts = conversion.convert_to_text_batches(list(files), batch_size)
        for file_extract in file_extracts:
            process_extract(file_extract, file_extract.path)
    end = time.time()
    click.echo(f"Elapsed time: {end - start} seconds.")


if __name__ == "__main__":
    cli()

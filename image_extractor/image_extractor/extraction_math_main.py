from pathlib import Path
from enum import StrEnum
import click
import time
import json
import os
from service.math_formula_extraction import (
    OpenAiMathFormulaConversion,
    VertexAiMathFormulaConversion,
    AnthropicMathFormulaConversion,
    GoogleMathFormulaConversion
)
from model.math_formula_extract import MathFormulaExtract

class Model(StrEnum):
    OPENAI = "openai"
    VERTEXAI = "vertexai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"

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
def extract_math_formulas(folder: str, model: str, extension: str):
    start = time.time()
    root = Path(folder)
    assert root.exists(), f"Path {root} does not exist."

    if model == Model.OPENAI.value:
        conversion = OpenAiMathFormulaConversion()
        model_type = os.getenv("OPENAI_MODEL", "").replace("gpt-", "")
    elif model == Model.VERTEXAI.value:
        conversion = VertexAiMathFormulaConversion()
        model_type = os.getenv("GEMINI_MODEL", "").replace("gemini-", "")
    elif model == Model.ANTHROPIC.value:
        conversion = AnthropicMathFormulaConversion()
        model_type = os.getenv("ANTHROPIC_MODEL", "").replace("claude-", "")
    elif model == Model.GOOGLE.value:
        conversion = GoogleMathFormulaConversion()
        model_type = os.getenv("GOOGLE_GENAI_MODEL", "").replace("gemini-", "")
    else:
        raise ValueError(f"Unsupported model type: {model}")

    files = root.rglob(f"*.{extension}")

    for f in files:
        output_file = f.parent / "results" / f"{model}_{model_type}_{f.stem}_math.json" 
        if output_file.exists():
            click.echo(f"File {f} was already processed.")
            continue
        click.echo(f"Processing file {f}")
        start_file = time.time()
        extract: MathFormulaExtract = conversion.convert_to_formula(f)
        elapsed_file = time.time() - start_file
        # Garante que o tempo está no resultado
        result = {
            "formula": extract.formula,
            "elapsed_time": extract.elapsed_time or elapsed_file
        }
        json_data = json.dumps(result, ensure_ascii=False, indent=2)
        output_file.write_text(json_data, encoding="utf-8")
        click.echo(f"Wrote {output_file}.")

    end = time.time()
    click.echo(f"Elapsed time: {end - start} seconds.")

if __name__ == "__main__":
    cli() 
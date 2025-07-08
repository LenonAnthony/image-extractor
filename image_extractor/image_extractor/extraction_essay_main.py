from pathlib import Path
from enum import StrEnum
import click
import time
import json
import os
from image_extractor.service.essay_evaluation import OpenAiEssayEvaluator, VertexAiEssayEvaluator, AnthropicEssayEvaluator
import pandas as pd

class Model(StrEnum):
    OPENAI = "openai"
    VERTEXAI = "vertexai"  
    ANTHROPIC = "anthropic"

@click.group()
def cli():
    pass

@cli.command()
@click.option(
    "--csv_file", 
    required=True, 
    help="Path to the CSV file containing essay data"
)
@click.option(
    "--model",
    default=Model.ANTHROPIC.value,
    prompt="The model you want to use",
    type=click.Choice([m.value for m in Model]),
    help="The model type to be used",
)
@click.option(
    "--output_dir", default="./output", help="Directory to save evaluation results"
)
@click.option(
    "--start_index", default=1, help="Index to start processing from in the CSV file"
)
@click.option(
    "--end_index", default=None, type=int, help="Index to end processing at in the CSV file"
)
def evaluate_essays(csv_file: str, model: str, output_dir: str, start_index: int, end_index: int):
    start = time.time()
    csv_path = Path(csv_file)
    assert csv_path.exists(), f"CSV file {csv_path} does not exist."
    
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(exist_ok=True, parents=True)
    
    evaluator = None
    if model == Model.OPENAI.value:
        evaluator = OpenAiEssayEvaluator()
        model_type = os.getenv("OPENAI_MODEL", "").replace("gpt-", "")
    elif model == Model.VERTEXAI.value:
        evaluator = VertexAiEssayEvaluator()
        model_type = os.getenv("GEMINI_MODEL", "").replace("gemini-", "")
    elif model == Model.ANTHROPIC.value:
        evaluator = AnthropicEssayEvaluator()
        model_type = os.getenv("ANTHROPIC_MODEL", "").replace("claude-", "")
    else:
        raise ValueError(f"Unsupported model type: {model}")

    df = pd.read_csv(csv_path)
    
    for idx, row in df.iloc[start_index:end_index].iterrows():
        output_file = output_dir_path / f"{model}_{model_type}_essay_{idx}.json"
        
        if output_file.exists():
            click.echo(f"Essay {idx} already evaluated. Skipping.")
            continue
            
        click.echo(f"Evaluating essay {idx}")
        essay_text = row["text"]
        prompt_text = row["prompt"]
        
        start_essay = time.time()
        try:
            result = evaluator.evaluate_essay(essay_text, prompt_text, idx)
            elapsed_essay = time.time() - start_essay
            
            result["elapsed"] = elapsed_essay
            
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
                
            click.echo(f"Evaluation for essay {idx} saved to {output_file}")
        except Exception as e:
            click.echo(f"Error evaluating essay {idx}: {e}")
    
    end = time.time()
    click.echo(f"Total elapsed time: {end - start} seconds")

if __name__ == "__main__":
    cli()

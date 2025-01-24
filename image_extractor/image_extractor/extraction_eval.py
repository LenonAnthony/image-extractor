import pandas as pd
from pathlib import Path
import json
from torcheval.metrics import WordErrorRate
from typing import Dict
import locale
import argparse

locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')

def load_openai_results(sample_dir: str) -> Dict[str, dict]:
    results = {}
    path = Path(sample_dir)
    for json_file in path.glob("openai_*.json"):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            file_num = json_file.stem.split("_")[-1]
            results[f"{file_num}.png"] = {
                "text": data["main_text"].lower().strip(),
                "elapsed": data.get("elapsed", 0.0)
            }
    return results

def load_vertexai_results(sample_dir: str) -> Dict[str, dict]:
    results = {}
    path = Path(sample_dir)
    for json_file in path.glob("vertexai_*.json"):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            file_num = json_file.stem.split("_")[-1]
            results[f"{file_num}.png"] = {
                "text": data["main_text"].lower().strip(),
                "elapsed": data.get("elapsed", 0.0)
            }
    return results

def load_ground_truth(csv_path: str) -> Dict[str, str]:
    df = pd.read_csv(csv_path)
    return {row["path"]: row["word"].lower().strip() for _, row in df.iterrows()}

def compare_results(api_results: Dict[str, dict], ground_truth: Dict[str, str]) -> pd.DataFrame:
    comparisons = []
    
    for path in ground_truth.keys():
        gt_text = ground_truth.get(path, "")
        result = api_results.get(path, {})
        extracted_text = result.get("text", "")
        elapsed = result.get("elapsed", 0.0)
        
        metric = WordErrorRate()
        metric.update([extracted_text], [gt_text])  
        wer = metric.compute().item() 
        
        exact_match = gt_text == extracted_text
        
        comparisons.append({
            "file": path,
            "ground_truth": gt_text,
            "extracted_text": extracted_text,
            "word_error_rate": wer,
            "exact_match": exact_match,
            "elapsed": elapsed
        })
    
    return pd.DataFrame(comparisons)

def generate_report(df: pd.DataFrame, output_path: str, model_name: str):
    total_files = len(df)
    exact_matches = df["exact_match"].sum()
    avg_wer = df["word_error_rate"].mean()
    total_elapsed = df["elapsed"].sum()
    
    df["word_error_rate"] = df["word_error_rate"].apply(lambda x: locale.format_string("%.2f", x))
    df["elapsed"] = df["elapsed"].apply(lambda x: locale.format_string("%.2f", x))
    
    df.to_csv(output_path, index=False)
    
    print(f"\nAnalysis Summary:")
    print(f"Total files analyzed: {total_files}")
    print(f"Exact matches: {exact_matches} ({(exact_matches/total_files)*100:.2f}%)")
    print(f"average WER: {avg_wer:.2f}")
    print(f"Total elapsed time: {locale.format_string('%.2f', total_elapsed)} seconds")

def main():
    parser = argparse.ArgumentParser(description='Generate report comparing the results of text extraction models')
    parser.add_argument('--model', required=True, choices=['openai', 'vertexai'],
                       help='Choice model (openai or vertexai)')
    args = parser.parse_args()
    
    sample_dir = "sample"
    words_csv = f"{sample_dir}/words.csv"
    output_csv = f"analysis_{args.model}.csv"

    if args.model == 'openai':
        results = load_openai_results(sample_dir)
    else:
        results = load_vertexai_results(sample_dir)
    
    ground_truth = load_ground_truth(words_csv)
    results_df = compare_results(results, ground_truth)
    
    generate_report(results_df, output_csv, args.model)

if __name__ == "__main__":
    main()
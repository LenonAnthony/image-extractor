import pandas as pd
from pathlib import Path
import json
from torcheval.metrics import WordErrorRate
from typing import Dict
import locale
import argparse
from Levenshtein import editops
import os
from dotenv import load_dotenv

load_dotenv()

locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')

def load_model_results(sample_dir: str, model: str, extension: str) -> Dict[str, dict]:
    results = {}
    path = Path(sample_dir)
    json_files = list(path.rglob(f"{model}_*.json"))
    processed_dirs = set()
    
    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            relative_path = json_file.parent.relative_to(path)
            file_num = json_file.stem.split("_")[-1]
            image_key = (relative_path / f"{file_num}.{extension}").as_posix()
            main_text = data["main_text"].lower().strip().replace('"""', '"').replace('"', '')
            results[image_key] = {
                "main_text": main_text,
                "elapsed": data.get("elapsed", 0.0)
            }
            processed_dirs.add(json_file.parent)
    
    all_subdirs = {d for d in path.rglob('*') if d.is_dir() and d != path}
    missing_dirs = all_subdirs - processed_dirs
    
    for dir_missing in missing_dirs:
        print(f"Warning: No JSON files for model '{model}' found in {dir_missing}")
        
    if not json_files:
        print(f"Warning: No JSON files found for model '{model}' in {sample_dir}")
    return results

def load_ground_truth(csv_path: str, sample_dir: str) -> Dict[str, str]:
    df = pd.read_csv(csv_path)
    df["word"] = df["word"].fillna("").astype(str).str.lower().str.strip()
    df["path"] = df["path"].apply(lambda p: p[len(sample_dir) + 1:] if p.startswith(sample_dir + os.sep) else p)
    return {row["path"]: row["word"] for _, row in df.iterrows()}

def calculate_cer(gt: str, extracted: str) -> float:
    if len(gt) == 0:
        return 1.0 if len(extracted) > 0 else 0.0
    edits = len(editops(gt, extracted))
    return edits / len(gt)

def calculate_char_metrics(gt: str, extracted: str) -> dict:
    if len(gt) == 0 and len(extracted) == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if len(gt) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    ops = editops(gt, extracted)
    insertions = sum(1 for op in ops if op[0] == 'insert')
    deletions = sum(1 for op in ops if op[0] == 'delete')
    replaces = sum(1 for op in ops if op[0] == 'replace')
    
    tp = len(gt) - (deletions + replaces)
    fp = insertions + replaces
    fn = deletions + replaces
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def compare_results(api_results: Dict[str, dict], ground_truth: Dict[str, str]) -> pd.DataFrame:
    comparisons = []
    for path in ground_truth.keys():
        gt_text = ground_truth.get(path, "")
        result = api_results.get(path, {})
        extracted_text = result.get("main_text", "")
        elapsed = result.get("elapsed", 0.0)
        
        metric_wer = WordErrorRate()
        metric_wer.update([extracted_text], [gt_text])
        wer = metric_wer.compute().item()
        
        cer = calculate_cer(gt_text, extracted_text)
        char_metrics = calculate_char_metrics(gt_text, extracted_text)
        
        comparisons.append({
            "file": path,
            "ground_truth": gt_text,
            "extracted_text": extracted_text,
            "word_error_rate": wer,
            "character_error_rate": cer,
            "precision": char_metrics["precision"], 
            "recall": char_metrics["recall"],
            "f1": char_metrics["f1"],
            "exact_match_char": (gt_text == extracted_text),
            "elapsed": elapsed
        })
    
    return pd.DataFrame(comparisons)

def generate_report(df: pd.DataFrame, output_path: str, model_name: str):
    total_files = len(df)
    exact_matches_char = df["exact_match_char"].sum()
    avg_wer = df["word_error_rate"].mean()
    avg_cer = df["character_error_rate"].mean()
    avg_precision = df["precision"].mean()
    avg_recall = df["recall"].mean()
    avg_f1 = df["f1"].mean()
    total_elapsed = df["elapsed"].sum()
    
    df["word_error_rate"] = df["word_error_rate"].apply(lambda x: locale.format_string("%.2f", x))
    df["character_error_rate"] = df["character_error_rate"].apply(lambda x: locale.format_string("%.2f", x))
    df["precision"] = df["precision"].apply(lambda x: locale.format_string("%.2f", x))
    df["recall"] = df["recall"].apply(lambda x: locale.format_string("%.2f", x))
    df["f1"] = df["f1"].apply(lambda x: locale.format_string("%.2f", x))
    df["elapsed"] = df["elapsed"].apply(lambda x: locale.format_string("%.2f", x))
    
    df.to_csv(output_path, sep=",", index=False)
    
    print(f"\nAnalysis Summary:")
    print(f"Total files analyzed: {total_files}")
    print(f"Exact character matches: {exact_matches_char} ({(exact_matches_char/total_files)*100:.2f}%)")
    print(f"Average WER: {avg_wer:.2f}")
    print(f"Average CER: {avg_cer:.2f}")
    print(f"Average Precision: {avg_precision:.2f}")
    print(f"Average Recall: {avg_recall:.2f}")
    print(f"Average F1-Score: {avg_f1:.2f}")
    print(f"Total elapsed time: {locale.format_string('%.2f', total_elapsed)} seconds")

def main():
    parser = argparse.ArgumentParser(description='Generate report comparing the results of text extraction models')
    parser.add_argument('--model', required=True, choices=['openai', 'vertexai'],
                       help='Model to evaluate (openai or vertexai)')
    parser.add_argument('--extension', default='png', choices=['png', 'jpg', 'jpeg', 'jfif'],
                       help='Image file extension to process (default: png)')
    parser.add_argument('--sample-dir', default='sample', help='Directory containing the sample data')
    parser.add_argument('--csv-file', default='words.csv', help='Name of the CSV file containing ground truth')
    args = parser.parse_args()
    
    sample_dir = args.sample_dir
    words_csv = os.path.join(sample_dir, args.csv_file)
    output_model = os.getenv("OPENAI_MODEL") if args.model == "openai" else os.getenv("GEMINI_MODEL")
    output_csv = f"{sample_dir}/analysis_{output_model}.csv"

    results = load_model_results(sample_dir, args.model, args.extension)
    ground_truth = load_ground_truth(words_csv, sample_dir)
    results_df = compare_results(results, ground_truth)
    
    generate_report(results_df, output_csv, args.model)

if __name__ == "__main__":
    main()
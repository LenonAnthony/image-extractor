import pandas as pd
from pathlib import Path
import json
from torcheval.metrics import WordErrorRate
from typing import Dict
import locale
import argparse
from Levenshtein import editops

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
        extracted_text = result.get("text", "")
        elapsed = result.get("elapsed", 0.0)
        
        # WER
        metric_wer = WordErrorRate()
        metric_wer.update([extracted_text], [gt_text])  
        wer = metric_wer.compute().item()
        
        #CER
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
    
    df.to_csv(output_path, index=False)
    
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
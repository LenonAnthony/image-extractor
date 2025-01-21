import pandas as pd
from pathlib import Path
import json
import difflib
from typing import Dict
import locale 

locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')

def load_openai_results(sample_dir: str) -> Dict[str, str]:
    """Load all OpenAI JSON results into a dictionary"""
    results = {}
    path = Path(sample_dir)
    for json_file in path.glob("openai_*.json"):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            file_num = json_file.stem.split("_")[1]
            results[f"{file_num}.png"] = data["main_text"].lower().strip()
    return results

def load_ground_truth(csv_path: str) -> Dict[str, str]:
    """Load ground truth from CSV file"""
    df = pd.read_csv(csv_path)
    return {row["path"]: row["word"].lower().strip() for _, row in df.iterrows()}

def calculate_similarity(str1: str, str2: str) -> float:
    """Calculate string similarity using difflib's SequenceMatcher"""
    return difflib.SequenceMatcher(None, str1, str2).ratio()

def compare_results(openai_results: Dict[str, str], ground_truth: Dict[str, str]) -> pd.DataFrame:
    """Compare OpenAI results with ground truth and return DataFrame with metrics"""
    comparisons = []
    
    for path in ground_truth.keys():
        gt_text = ground_truth.get(path, "")
        extracted_text = openai_results.get(path, "")
        
        similarity = calculate_similarity(gt_text, extracted_text)
        exact_match = gt_text == extracted_text
        
        comparisons.append({
            "file": path,
            "ground_truth": gt_text,
            "extracted_text": extracted_text,
            "similarity": similarity,
            "exact_match": exact_match
        })
    
    return pd.DataFrame(comparisons)

def generate_report(df: pd.DataFrame, output_path: str):
    """Generate CSV report with comparison metrics"""
    total_files = len(df)
    exact_matches = df["exact_match"].sum()
    avg_similarity = df["similarity"].mean()
    df["similarity"] = df["similarity"].apply(lambda x: locale.format_string("%.2f", x))
    df.to_csv(output_path, index=False)
    
    print(f"\nAnalysis Summary:")
    print(f"Total files analyzed: {total_files}")
    print(f"Exact matches: {exact_matches} ({(exact_matches/total_files)*100:.2f}%)")
    print(f"Average similarity: {avg_similarity:.2f}")

def main():
    sample_dir = "sample"
    words_csv = f"{sample_dir}/words.csv"
    output_csv = "extraction_analysis.csv"
    
    openai_results = load_openai_results(sample_dir)
    ground_truth = load_ground_truth(words_csv)
    
    results_df = compare_results(openai_results, ground_truth)
    generate_report(results_df, output_csv)

if __name__ == "__main__":
    main()
import pandas as pd
from Levenshtein import editops

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
    
    return {"precision": precision, "recall": recall, "f1": f1}

def generate_report(input_csv: str, output_csv: str):
    df = pd.read_csv(input_csv)
    
    df["word"] = df["word"].fillna("").astype(str).str.lower().str.strip()
    df["pred"] = df["pred"].fillna("").astype(str).str.lower().str.strip()
    
    results = []
    for _, row in df.iterrows():
        metrics = calculate_char_metrics(row["word"], row["pred"])
        results.append({
            "exact_match": row["word"] == row["pred"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"]
        })
    
    report_df = pd.concat([df, pd.DataFrame(results)], axis=1)

    report_df.to_csv(output_csv, index=False)
    
    total = len(report_df)
    exact_matches = report_df["exact_match"].sum()
    
    print(f"\nanalysis:")
    print(f"len words: {total}")
    print(f"exact matches (CER): {exact_matches} ({exact_matches/total:.2%})")
    print(f"avg precision: {report_df['precision'].mean():.2f}")
    print(f"avg recall: {report_df['recall'].mean():.2f}")
    print(f"avg f1: {report_df['f1'].mean():.2f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='A script to process input and output CSV files.')
    parser.add_argument('--input', required=True, help='Input CSV file path')
    parser.add_argument('--output', required=True, help='Output CSV file path')
    args = parser.parse_args()
    
    generate_report(args.input, args.output)
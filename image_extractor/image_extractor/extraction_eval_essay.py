import pandas as pd
from pathlib import Path
import json
from typing import Dict
import locale
import argparse
import os
from dotenv import load_dotenv
from Levenshtein import editops

load_dotenv()

locale.setlocale(locale.LC_ALL, "pt_BR.UTF-8")

def load_model_results(sample_dir: str, model: str, extension: str) -> Dict[str, dict]:
    results = {}
    path = Path(sample_dir)

    json_files = list(path.rglob(f"imgs/{model}_*.json"))
    processed_dirs = set()

    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

            relative_path = json_file.parent.relative_to(path)

            file_num = json_file.stem.split("_")[-1]

            image_key = (relative_path / f"{file_num}.{extension}").as_posix()

            main_text = (
                data["main_text"].lower().strip().replace('"""', '"').replace('"', "")
            )
            results[image_key] = {
                "main_text": main_text,
                "elapsed": data.get("elapsed", 0.0),
            }
            processed_dirs.add(json_file.parent)

    all_imgs_dirs = {d for d in path.rglob("imgs") if d.is_dir()}
    missing_imgs_dirs = all_imgs_dirs - processed_dirs

    for dir_missing in missing_imgs_dirs:
        lote_dir = dir_missing.parent.name
        print(f"Warning: No JSONs for model '{model}' in {lote_dir}/{dir_missing.name}")

    return results


def load_ground_truth(csv_path: str) -> Dict[str, str]:
    df = pd.read_csv(csv_path)
    df["word"] = df["word"].fillna("").astype(str).str.lower().str.strip()
    df["path"] = (
        df["path"]
        .str.replace(r"^image_to_text/", "", regex=True)
        .str.replace(os.sep, "/", regex=False)
    )

    return {row["path"]: row["word"] for _, row in df.iterrows()}


def calculate_char_metrics(gt: str, extracted: str) -> dict:
    if len(gt) == 0 and len(extracted) == 0:
        return {
            "precision": 1.0, 
            "recall": 1.0, 
            "f1": 1.0, 
            "cer": 0,
            "total_chars": 0
        }
    
    if len(gt) == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "cer": len(extracted),
            "total_chars": 0
        }

    ops = editops(gt, extracted)
    deletions = sum(1 for op in ops if op[0] == "delete")
    replaces = sum(1 for op in ops if op[0] == "replace")
    insertions = sum(1 for op in ops if op[0] == "insert")

    cer = deletions + replaces
    total_chars = len(gt)

    tp = total_chars - (deletions + replaces)
    fp = insertions + replaces
    fn = deletions + replaces

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "cer": cer,
        "total_chars": total_chars  
    }


def calculate_word_metrics(gt: str, extracted: str) -> dict:
    gt_words = gt.split()
    extracted_words = extracted.split()

    if not gt_words:
        return {"wer": 1.0, "correct_words": 0, "total_words": 0, "total_errors": 0}

    ops = editops(gt_words, extracted_words)
    errors = len(ops)
    total_words = len(gt_words)
    correct = total_words - (errors - sum(1 for op in ops if op[0] == "insert"))

    return {
        "wer": errors / total_words,
        "correct_words": correct,
        "total_words": total_words,
        "total_errors": errors,
    }


def compare_results(
    api_results: Dict[str, dict], ground_truth: Dict[str, str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    comparisons = []
    word_errors = []

    for path in ground_truth.keys():
        normalized_path = path.replace(os.sep, "/")
        gt_text = ground_truth.get(normalized_path, "")
        result = api_results.get(normalized_path, {})
        extracted_text = result.get("main_text", "").strip()
        elapsed = result.get("elapsed", 0.0)

        word_metrics = calculate_word_metrics(gt_text, extracted_text)
        char_metrics = calculate_char_metrics(gt_text, extracted_text)

        gt_words = gt_text.split()
        extracted_words = extracted_text.split()
        ops = editops(gt_words, extracted_words)

        for op in ops:
            error_type = op[0]
            gt_pos = op[1]
            extracted_pos = op[2]

            error_details = {
                "file": normalized_path,
                "error_type": error_type,
                "gt_word": (
                    gt_words[gt_pos] if error_type in ["delete", "replace"] else None
                ),
                "extracted_word": (
                    extracted_words[extracted_pos]
                    if error_type in ["insert", "replace"]
                    else None
                ),
            }
            word_errors.append(error_details)

        comparisons.append(
            {
                "file": normalized_path,
                "ground_truth": gt_text,
                "extracted_text": extracted_text,
                "wer": word_metrics["wer"],
                "correct_words": word_metrics["correct_words"],
                "total_words": word_metrics["total_words"],
                "total_errors": word_metrics["total_errors"],
                "precision": char_metrics["precision"],
                "recall": char_metrics["recall"],
                "f1": char_metrics["f1"],
                "cer": char_metrics["cer"],
                "total_chars": char_metrics["total_chars"],  # Novo campo adicionado
                "elapsed": elapsed,
            }
        )

    return pd.DataFrame(comparisons), pd.DataFrame(word_errors)

def generate_report(
    df: pd.DataFrame, errors_df: pd.DataFrame, output_path: str, model_name: str
):
    total_files = len(df)
    total_words = df["total_words"].sum()
    total_errors = df["total_errors"].sum()
    total_correct = df["correct_words"].sum()
    avg_wer = df["wer"].mean()
    total_elapsed = df["elapsed"].sum()

    # Novas métricas de caracteres
    total_cer = df["cer"].sum()
    total_chars = df["total_chars"].sum() 
    cer_rate = (total_cer / total_chars) if total_chars > 0 else 0

    avg_precision = df["precision"].mean()
    avg_recall = df["recall"].mean()
    avg_f1 = df["f1"].mean()

    df["wer"] = df["wer"].apply(lambda x: locale.format_string("%.2f", x))
    df["elapsed"] = df["elapsed"].apply(lambda x: locale.format_string("%.2f", x))

    errors_output = output_path.replace(".csv", "_errors.csv")
    errors_df.to_csv(errors_output, index=False)
    df.to_csv(output_path, sep=",", index=False)

    print(f"\nAnalysis Summary:")
    print(f"Total files analyzed: {total_files}")
    print(f"Total words: {total_words}")
    print(f"Correct words: {total_correct} ({(total_correct/total_words)*100:.2f}%)")
    print(f"Total word errors: {total_errors}")
    print(f"Average WER: {avg_wer:.2%}")
    print(f"Total character errors (CER): {total_cer}")
    print(f"Total characters in ground truth: {total_chars}")
    print(f"CER Rate: {cer_rate:.2%}")  
    print(f"Average CER per file: {total_cer/total_files:.2f}")
    print(f"Average Precision: {avg_precision:.2%}")
    print(f"Average Recall: {avg_recall:.2%}")
    print(f"Average F1: {avg_f1:.2%}")
    print(f"Total elapsed time: {locale.format_string('%.2f', total_elapsed)} seconds")


def main():
    parser = argparse.ArgumentParser(
        description="Generate report comparing the results of text extraction models"
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["openai", "vertexai", "google_vision", "anthropic"],
        help="Model to evaluate (openai, vertexai, google_vision, anthropic)",
    )
    parser.add_argument(
        "--extension",
        default="png",
        choices=["png", "jpg", "jpeg", "jfif"],
        help="Image file extension to process (default: png)",
    )
    parser.add_argument(
        "--sample-dir", default="sample", help="Directory containing the sample data"
    )
    parser.add_argument(
        "--csv-file",
        default="words.csv",
        help="Name of the CSV file containing ground truth",
    )
    args = parser.parse_args()

    sample_dir = args.sample_dir
    words_csv = os.path.join(sample_dir, args.csv_file)
    output_model = (
        os.getenv("OPENAI_MODEL")
        if args.model == "openai"
        else (
            os.getenv("GEMINI_MODEL")
            if args.model == "gemini"
            else (
                os.getenv("ANTHROPIC_MODEL")
                if args.model == "anthropic"
                else os.getenv("OPENAI_MODEL")
            )
        )  
    )
    output_csv = f"{sample_dir}/analysis_{output_model}.csv"

    results = load_model_results(sample_dir, args.model, args.extension)
    ground_truth = load_ground_truth(words_csv)
    results_df, results_error = compare_results(results, ground_truth)

    generate_report(results_df, results_error, output_csv, args.model)


if __name__ == "__main__":
    main()

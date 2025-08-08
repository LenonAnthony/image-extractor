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

try:
    locale.setlocale(locale.LC_ALL, "pt_BR.UTF-8")
except locale.Error:
    try:
        locale.setlocale(locale.LC_ALL, "C.UTF-8")
    except locale.Error:
        locale.setlocale(locale.LC_ALL, "")


def load_model_results(sample_dir: str, model: str, extension: str) -> Dict[str, dict]:
    results = {}
    path = Path(sample_dir)
    # Determinar o nome do modelo baseado no parâmetro
    if model == "huggingface":
        model_name = "huggingface_CEIA-UFG_Gemma-3-Gaia-PT-BR-4b-it"
    else:
        model_name = model
    json_files = list(path.rglob(f"{model_name}_*.json"))
    processed_dirs = set()

    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Extrair o número do arquivo (última parte do nome)
            file_num = json_file.stem.split("_")[-1]
            # Criar a chave da imagem baseada no caminho relativo
            relative_path = json_file.parent.relative_to(path)
            # Usar .jpeg como extensão padrão para as imagens
            image_key = f"{relative_path}/{file_num}.jpeg"
            main_text = (
                data["main_text"].lower().strip().replace('"""', '"').replace('"', "")
            )
            results[image_key] = {
                "main_text": main_text,
                "elapsed": data.get("elapsed", 0.0),
            }
            processed_dirs.add(json_file.parent)

    all_subdirs = {d for d in path.rglob("*") if d.is_dir() and d != path}
    missing_dirs = all_subdirs - processed_dirs

    for dir_missing in missing_dirs:
        print(f"Warning: No JSON files for model '{model_name}' found in {dir_missing}")

    if not json_files:
        print(f"Warning: No JSON files found for model '{model_name}' in {sample_dir}")
    return results


def load_ground_truth(csv_path: str, sample_dir: str) -> Dict[str, str]:
    df = pd.read_csv(csv_path)
    df["word"] = df["word"].fillna("").astype(str).str.lower().str.strip()
    df["path"] = df["path"].apply(
        lambda p: p[len(sample_dir) + 1 :] if p.startswith(sample_dir + os.sep) else p
    )
    return {row["path"]: row["word"] for _, row in df.iterrows()}


def calculate_char_metrics(gt: str, extracted: str) -> dict:
    if len(gt) == 0 and len(extracted) == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if len(gt) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    ops = editops(gt, extracted)
    insertions = sum(1 for op in ops if op[0] == "insert")
    deletions = sum(1 for op in ops if op[0] == "delete")
    replaces = sum(1 for op in ops if op[0] == "replace")

    tp = len(gt) - (deletions + replaces)
    fp = insertions + replaces
    fn = deletions + replaces

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {"precision": precision, "recall": recall, "f1": f1}


def compare_results(
    api_results: Dict[str, dict], ground_truth: Dict[str, str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    comparisons = []
    word_errors = []

    for path in ground_truth.keys():
        gt_text = ground_truth.get(path, "")
        result = api_results.get(path, {})
        extracted_text = result.get("main_text", "").strip()
        elapsed = result.get("elapsed", 0.0)

        char_metrics = calculate_char_metrics(gt_text, extracted_text)

        # Coletar erros por palavra
        gt_words = gt_text.split()
        extracted_words = extracted_text.split()
        ops = editops(gt_words, extracted_words)

        for op in ops:
            error_type = op[0]
            gt_pos = op[1]
            extracted_pos = op[2]

            error_details = {
                "file": path,
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
                "file": path,
                "ground_truth": gt_text,
                "extracted_text": extracted_text,
                "precision": char_metrics["precision"],
                "recall": char_metrics["recall"],
                "f1": char_metrics["f1"],
                "exact_match_char": (gt_text == extracted_text),
                "elapsed": elapsed,
            }
        )

    return pd.DataFrame(comparisons), pd.DataFrame(word_errors)


def generate_report(
    df: pd.DataFrame, errors_df: pd.DataFrame, output_path: str, model_name: str
):
    total_files = len(df)
    total_elapsed = df["elapsed"].sum()
    
    # Calcular métricas de palavras
    total_words = 0
    correct_words = 0
    total_word_errors = 0
    
    total_characters_gt = 0
    total_cer = 0
    
    for _, row in df.iterrows():
        gt_text = row["ground_truth"]
        extracted_text = row["extracted_text"]
        
        gt_words = gt_text.split()
        extracted_words = extracted_text.split()
        total_words += len(gt_words)
        # Calcular palavras corretas e erros
        ops = editops(gt_words, extracted_words)
        word_errors = len(ops)
        total_word_errors += word_errors
        correct_words += len(gt_words) - word_errors
        
        # Métricas de caracteres
        total_characters_gt += len(gt_text)
        char_ops = editops(gt_text, extracted_text)
        total_cer += len(char_ops)
    
    # Calcular métricas médias
    avg_precision = df["precision"].mean()
    avg_recall = df["recall"].mean()
    avg_f1 = df["f1"].mean()
    
    # Calcular WER e CER
    avg_wer = (total_word_errors / total_words * 100) if total_words > 0 else 0
    cer_rate = (total_cer / total_characters_gt * 100) if total_characters_gt > 0 else 0
    avg_cer_per_file = total_cer / total_files if total_files > 0 else 0
    
    # Calcular porcentagem de palavras corretas
    correct_words_percentage = (correct_words / total_words * 100) if total_words > 0 else 0

    word_error_counts = errors_df[errors_df["error_type"].isin(["replace", "delete"])]
    word_error_counts = (
        word_error_counts.groupby("gt_word").size().reset_index(name="count")
    )
    word_error_counts = word_error_counts.sort_values("count", ascending=False)

    word_errors_csv = output_path.replace(".csv", "_word_errors.csv")
    word_error_counts.to_csv(word_errors_csv, index=False)

    print(f"\nAnalysis Summary:")
    print(f"Total files analyzed: {total_files}")
    print(f"Total words: {total_words}")
    print(f"Correct words: {correct_words} ({correct_words_percentage:.2f}%)")
    print(f"Total word errors: {total_word_errors}")
    print(f"Average WER (Word Error Rate): {avg_wer:.2f}%")
    print(f"Total character errors (CER): {total_cer}")
    print(f"Total characters in ground truth: {total_characters_gt}")
    print(f"CER Rate: {cer_rate:.2f}%")
    print(f"Average CER per file: {avg_cer_per_file:.2f}")
    print(f"Average Precision: {avg_precision:.2f}")
    print(f"Average Recall: {avg_recall:.2f}")
    print(f"Average F1: {avg_f1:.2f}")
    print(f"Total elapsed time: {locale.format_string('%.2f', total_elapsed)} seconds")

    print("\nTop 10:")
    if not word_error_counts.empty:
        top_errors = word_error_counts.head(10)
        for _, row in top_errors.iterrows():
            print(f"Word '{row['gt_word']}': {row['count']} errors")
    else:
        print("No word errors found.")

    df["precision"] = df["precision"].apply(lambda x: locale.format_string("%.2f", x))
    df["recall"] = df["recall"].apply(lambda x: locale.format_string("%.2f", x))
    df["f1"] = df["f1"].apply(lambda x: locale.format_string("%.2f", x))
    df["elapsed"] = df["elapsed"].apply(lambda x: locale.format_string("%.2f", x))
    df.to_csv(output_path, sep=",", index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Generate report comparing the results of text extraction models"
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["openai", "vertexai", "google_vision", "anthropic", "ollama", "huggingface"],
        help="Model to evaluate (openai, vertexai, google_vision, anthropic, ollama, or huggingface)",
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
    # Se o CSV não existir no diretório especificado, tentar no diretório pai
    if not os.path.exists(words_csv):
        words_csv = os.path.join(os.path.dirname(sample_dir), args.csv_file)
    output_model = (
        os.getenv("OPENAI_MODEL")
        if args.model == "openai"
        else (
            os.getenv("GEMINI_MODEL")
            if args.model == "vertexai"
            else (
                os.getenv("ANTHROPIC_MODEL")
                if args.model == "anthropic"
                else (
                    os.getenv("OLLAMA_MODEL", "minicpm-v:8b").replace(":", "_")
                    if args.model == "ollama"
                    else os.getenv("OPENAI_MODEL")
                )
            )
        )  
    )
    output_csv = f"{sample_dir}/analysis_{output_model}.csv"

    results = load_model_results(sample_dir, args.model, args.extension)
    ground_truth = load_ground_truth(words_csv, sample_dir)
    results_df, errors_df = compare_results(results, ground_truth)

    generate_report(results_df, errors_df, output_csv, args.model)


if __name__ == "__main__":
    main()

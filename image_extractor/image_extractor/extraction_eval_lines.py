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
    json_files = list(path.rglob(f"*/{model}_*.json"))
    processed_dirs = set()

    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            parent_dir = json_file.parent.name
            file_num = json_file.stem.split("_")[-1]
            image_key = f"{parent_dir}/{file_num}.{extension}"
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
        print(f"Warning: No JSON files for model '{model}' found in {dir_missing}")

    if not json_files:
        print(f"Warning: No JSON files found for model '{model}' in {sample_dir}")
    return results


def load_ground_truth(csv_path: str) -> Dict[str, str]:
    df = pd.read_csv(csv_path)
    df["text"] = df["text"].fillna("").astype(str).str.lower().str.strip()
    df["path"] = df["path"].astype(str).str.strip()
    return {row["path"]: row["text"] for _, row in df.iterrows()}


def calculate_char_metrics(gt: str, extracted: str) -> dict:
    if len(gt) == 0 and len(extracted) == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "cer": 0, "total_chars": 0}
    if len(gt) == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "cer": len(extracted),
            "total_chars": 0,
        }

    ops = editops(gt, extracted)
    insertions = sum(1 for op in ops if op[0] == "insert")
    deletions = sum(1 for op in ops if op[0] == "delete")
    replaces = sum(1 for op in ops if op[0] == "replace")

    cer = (insertions + deletions + replaces) / len(gt) if len(gt) > 0 else 0
    total_chars = len(gt)

    tp = total_chars - (deletions + replaces)
    fp = insertions + replaces
    fn = deletions + replaces

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "cer": cer,
        "total_chars": total_chars,
    }


def calculate_word_metrics(gt: str, extracted: str) -> dict:
    gt_words = gt.split()
    extracted_words = extracted.split()

    if not gt_words:
        return {"wer": 1.0, "correct_words": 0, "total_words": 0}

    ops = editops(gt_words, extracted_words)
    errors = len(ops)
    total_words = len(gt_words)
    wer = errors / total_words if total_words > 0 else 0

    return {
        "wer": wer,
        "correct_words": total_words - errors,
        "total_words": total_words,
    }


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
        word_metrics = calculate_word_metrics(gt_text, extracted_text)

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
                "wer": word_metrics["wer"],
                "correct_words": word_metrics["correct_words"],
                "total_words": word_metrics["total_words"],
                "precision": char_metrics["precision"],
                "recall": char_metrics["recall"],
                "f1": char_metrics["f1"],
                "cer": char_metrics["cer"],
                "total_chars": char_metrics["total_chars"],
                "exact_match_char": (gt_text == extracted_text),
                "elapsed": elapsed,
            }
        )

    return pd.DataFrame(comparisons), pd.DataFrame(word_errors)


def generate_report(
    df: pd.DataFrame, errors_df: pd.DataFrame, output_path: str, model_name: str
):
    total_files = len(df)
    exact_matches_char = df["exact_match_char"].sum()
    total_words = df["total_words"].sum()
    correct_words = df["correct_words"].sum()
    avg_wer = df["wer"].mean()
    total_chars = df["total_chars"].sum()
    total_cer = df["cer"].sum() * total_files
    cer_rate = df["cer"].mean()
    avg_precision = df["precision"].mean()
    avg_recall = df["recall"].mean()
    avg_f1 = df["f1"].mean()
    total_elapsed = df["elapsed"].sum()

    df["precision"] = df["precision"].apply(lambda x: locale.format_string("%.2f", x))
    df["recall"] = df["recall"].apply(lambda x: locale.format_string("%.2f", x))
    df["f1"] = df["f1"].apply(lambda x: locale.format_string("%.2f", x))
    df["wer"] = df["wer"].apply(lambda x: locale.format_string("%.2f", x))
    df["cer"] = df["cer"].apply(lambda x: locale.format_string("%.2f", x))
    df["elapsed"] = df["elapsed"].apply(lambda x: locale.format_string("%.2f", x))

    errors_output = output_path.replace(".csv", "_errors.csv")
    errors_df.to_csv(errors_output, index=False)
    df.to_csv(output_path, sep=",", index=False)

    print(f"\nResumo da Análise ({model_name}):")
    print(f"Total de arquivos analisados: {total_files}")
    print(
        f"Acerto exato de caracteres: {exact_matches_char} ({(exact_matches_char/total_files)*100:.2f}%)"
    )
    print(f"Total de palavras: {total_words}")
    print(
        f"Palavras corretas: {correct_words} ({(correct_words/total_words)*100:.2f}%)"
    )
    print(f"Média WER: {avg_wer:.2%}")
    print(f"Total de caracteres no ground truth: {total_chars}")
    print(f"CER absoluto: {cer_rate * total_chars:.2f}")
    print(f"Média CER por arquivo: {cer_rate:.2%}")
    print(f"Média Precision: {avg_precision:.2%}")
    print(f"Média Recall: {avg_recall:.2%}")
    print(f"Média F1-Score: {avg_f1:.2%}")
    print(
        f"Tempo total de execução: {locale.format_string('%.2f', total_elapsed)} segundos"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Gerar relatório comparando resultados de extração de texto"
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["openai", "vertexai", "google_vision", "anthropic"],
        help="Modelo a avaliar (openai, vertexai, google_vision, anthropic)",
    )
    parser.add_argument(
        "--extension",
        default="jpg",
        choices=["png", "jpg", "jpeg", "jfif"],
        help="Extensão dos arquivos de imagem (padrão: jpg)",
    )
    parser.add_argument(
        "--sample-dir",
        default="dataset-15-test",
        help="Diretório contendo os dados de amostra",
    )
    parser.add_argument(
        "--csv-file",
        default="words.csv",
        help="Nome do arquivo CSV com o ground truth",
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
    results_df, errors_df = compare_results(results, ground_truth)

    generate_report(results_df, errors_df, output_csv, args.model)


if __name__ == "__main__":
    main()

import pandas as pd
from pathlib import Path
import json
from typing import Dict, List, Tuple
import argparse
import os
from dotenv import load_dotenv

load_dotenv()

def load_model_results(sample_dir: str, model: str) -> Dict[str, List[dict]]:
    """Load the answer sheet extraction results from JSON files."""
    results = {}
    path = Path(sample_dir)
    results_dir = path / "results"
    
    if not results_dir.exists():
        print(f"Warning: Results directory not found at {results_dir}")
        return results
    
    if model == "openai":
        model_prefix = "openai_"
        model_type = os.getenv("OPENAI_MODEL", "").replace("gpt-", "")
    elif model == "vertexai":
        model_prefix = "vertexai_"
        model_type = os.getenv("GEMINI_MODEL", "").replace("gemini-", "")
    elif model == "anthropic":
        model_prefix = "anthropic_"
        model_type = os.getenv("ANTHROPIC_MODEL", "").replace("claude-", "")
    else:
        model_prefix = f"{model}_"
        model_type = ""
        
    search_pattern = f"{model_prefix}{model_type}_*.json"
    json_files = list(results_dir.glob(search_pattern))
    
    if not json_files:
        print(f"Warning: No JSON files found matching pattern {search_pattern} in {results_dir}")
        return results
    
    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            sheet_id = data.get("sheet_id", json_file.stem.split("_")[-1])
            results[sheet_id] = {
                "answers": data.get("answers", []),
                "total_questions": data.get("total_questions", 0),
                "elapsed": data.get("elapsed", 0.0),
            }
            
    return results


def load_ground_truth(csv_path: str) -> Dict[str, Dict[int, str]]:
    """Load ground truth data from CSV file."""
    df = pd.read_csv(csv_path)
    
    # Check required columns exist
    required_cols = ["sheet_id", "question_number", "correct_option"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV file is missing required columns: {missing_cols}")
    
    # Group by sheet_id
    result = {}
    for _, row in df.iterrows():
        sheet_id = str(row["sheet_id"])
        question_number = int(row["question_number"])
        correct_option = str(row["correct_option"]).upper()
        
        if sheet_id not in result:
            result[sheet_id] = {}
            
        result[sheet_id][question_number] = correct_option
        
    return result


def calculate_metrics(extracted_results: Dict[str, List[dict]], 
                      ground_truth: Dict[str, Dict[int, str]]) -> Tuple[pd.DataFrame, Dict]:
    """Calculate evaluation metrics for answer sheet extraction."""
    eval_data = []
    summary = {
        "total_sheets": 0,
        "total_questions": 0,
        "correct_answers": 0,
        "accuracy": 0.0,
        "avg_confidence": 0.0,
        "total_time": 0.0,
        "questions_by_confidence": {
            "high": {"total": 0, "correct": 0, "accuracy": 0.0},
            "medium": {"total": 0, "correct": 0, "accuracy": 0.0},
            "low": {"total": 0, "correct": 0, "accuracy": 0.0},
        }
    }
    
    # Process each answer sheet
    for sheet_id, gt_answers in ground_truth.items():
        if sheet_id not in extracted_results:
            print(f"Warning: No extraction results found for sheet {sheet_id}")
            continue
            
        extracted_data = extracted_results[sheet_id]
        extracted_answers = {a["question_number"]: a for a in extracted_data["answers"]}
        
        summary["total_sheets"] += 1
        summary["total_time"] += extracted_data.get("elapsed", 0)
        
        # Evaluate each question
        for question_num, correct_opt in gt_answers.items():
            summary["total_questions"] += 1
            
            if question_num not in extracted_answers:
                # Question wasn't detected
                eval_data.append({
                    "sheet_id": sheet_id,
                    "question_number": question_num,
                    "ground_truth": correct_opt,
                    "extracted": "missing",
                    "is_correct": False,
                    "confidence": 0.0,
                    "confidence_level": "missing"
                })
                continue
                
            extracted = extracted_answers[question_num]["marked_option"].upper()
            confidence = extracted_answers[question_num]["confidence"]
            is_correct = (extracted == correct_opt)
            
            # Determine confidence level
            if confidence >= 0.8:
                confidence_level = "high"
            elif confidence >= 0.5:
                confidence_level = "medium"
            else:
                confidence_level = "low"
                
            # Update confidence level stats
            summary["questions_by_confidence"][confidence_level]["total"] += 1
            summary["avg_confidence"] += confidence
            
            if is_correct:
                summary["correct_answers"] += 1
                summary["questions_by_confidence"][confidence_level]["correct"] += 1
                
            eval_data.append({
                "sheet_id": sheet_id,
                "question_number": question_num,
                "ground_truth": correct_opt,
                "extracted": extracted,
                "is_correct": is_correct,
                "confidence": confidence,
                "confidence_level": confidence_level
            })
    
    # Calculate summary statistics
    if summary["total_questions"] > 0:
        summary["accuracy"] = summary["correct_answers"] / summary["total_questions"]
        summary["avg_confidence"] /= summary["total_questions"]
        
        # Calculate accuracy by confidence level
        for level in ["high", "medium", "low"]:
            level_total = summary["questions_by_confidence"][level]["total"]
            level_correct = summary["questions_by_confidence"][level]["correct"]
            summary["questions_by_confidence"][level]["accuracy"] = (
                level_correct / level_total if level_total > 0 else 0.0
            )
            
    return pd.DataFrame(eval_data), summary


def generate_report(eval_df: pd.DataFrame, summary: Dict, output_path: str, model_name: str):
    """Generate and save evaluation report."""
    # Save detailed results to CSV
    eval_df.to_csv(output_path, index=False)
    
    # Save summary report to JSON
    summary_path = output_path.replace(".csv", "_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    
    # Print summary report
    print("\n===== ANSWER SHEET EXTRACTION EVALUATION =====")
    print(f"Model: {model_name}")
    print(f"Total answer sheets: {summary['total_sheets']}")
    print(f"Total questions: {summary['total_questions']}")
    print(f"Correct answers: {summary['correct_answers']}")
    print(f"Overall accuracy: {summary['accuracy']:.2%}")
    print(f"Average confidence: {summary['avg_confidence']:.2f}")
    print(f"Total processing time: {summary['total_time']:.2f} seconds")
    
    print("\nAccuracy by confidence level:")
    for level in ["high", "medium", "low"]:
        level_data = summary["questions_by_confidence"][level]
        level_total = level_data["total"]
        if level_total > 0:
            print(f"  {level.title()} confidence ({level_total} questions): {level_data['accuracy']:.2%} accuracy")
            
    print("\nConfusion matrix:")
    # Create confusion matrix
    confusion_matrix = pd.crosstab(
        eval_df["ground_truth"], 
        eval_df["extracted"],
        rownames=["Actual"], 
        colnames=["Predicted"],
        margins=True
    )
    print(confusion_matrix)
    

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate answer sheet extraction results"
    )
    parser.add_argument(
        "--model", 
        required=True,
        choices=["openai", "vertexai", "anthropic"], 
        help="Model used for extraction"
    )
    parser.add_argument(
        "--sample-dir", 
        default="answer_sheets", 
        help="Directory containing the sample data"
    )
    parser.add_argument(
        "--csv-file",
        default="ground_truth.csv",
        help="Name of the CSV file containing ground truth"
    )
    
    args = parser.parse_args()
    
    sample_dir = args.sample_dir
    ground_truth_csv = os.path.join(sample_dir, args.csv_file)
    
    # Determine model name for output
    if args.model == "openai":
        model_name = os.getenv("OPENAI_MODEL", "openai")
    elif args.model == "vertexai":
        model_name = os.getenv("GEMINI_MODEL", "vertexai")
    elif args.model == "anthropic":
        model_name = os.getenv("ANTHROPIC_MODEL", "anthropic")
    else:
        model_name = args.model
        
    output_csv = f"{sample_dir}/eval_{model_name.replace('-', '_')}.csv"
    
    # Load data
    extracted_results = load_model_results(sample_dir, args.model)
    if not extracted_results:
        print(f"No extraction results found for model {args.model}. Exiting.")
        return
        
    ground_truth = load_ground_truth(ground_truth_csv)
    if not ground_truth:
        print(f"No ground truth data found in {ground_truth_csv}. Exiting.")
        return
        
    # Calculate metrics and generate report
    eval_df, summary = calculate_metrics(extracted_results, ground_truth)
    generate_report(eval_df, summary, output_csv, model_name)
    
    print(f"\nDetailed evaluation saved to {output_csv}")
    print(f"Summary report saved to {output_csv.replace('.csv', '_summary.json')}")


if __name__ == "__main__":
    main()
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import click
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, List, Any, Tuple
import re
import math

def load_original_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df

def extract_index(filename: str) -> int:
    # Updated pattern to handle different filename formats
    match = re.search(r"essay_(\d+)\.json", filename)
    if not match:
        # Try alternative pattern
        match = re.search(r"_(\d+)\.json", filename)
    return int(match.group(1)) if match else -1

def load_prediction_files(predictions_dir: str, model_filter: str = None) -> List[Dict[str, Any]]:
    predictions_path = Path(predictions_dir).resolve()
    print(f"Looking for prediction files in: {predictions_path}")
    
    # List a few files in the directory to help with debugging
    try:
        all_files = list(predictions_path.glob("*"))
        if all_files:
            print(f"Found {len(all_files)} files in the directory. First few: {all_files[:3]}")
        else:
            print(f"No files found in {predictions_path}")
    except Exception as e:
        print(f"Error listing directory: {e}")
    
    if model_filter:
        prediction_files = [f for f in predictions_path.glob("*.json") if model_filter.lower() in f.name.lower()]
        print(f"Using model filter: {model_filter}, found {len(prediction_files)} matching files")
    else:
        prediction_files = list(predictions_path.glob("*.json"))
        print(f"No model filter, found {len(prediction_files)} JSON files")

    if prediction_files:
        print(f"First few prediction files: {[f.name for f in prediction_files[:3]]}")
    
    prediction_files = sorted(prediction_files, key=lambda f: extract_index(f.name))
    predictions = []
    for file in prediction_files:
        with open(file, 'r', encoding='utf-8') as f:
            try:
                prediction = json.load(f)
                prediction['filename'] = file.name
                predictions.append(prediction)
            except json.JSONDecodeError:
                print(f"Error loading JSON from {file}")
    return predictions

def match_predictions_with_originals(predictions: List[Dict[str, Any]], originals: pd.DataFrame) -> List[Dict[str, Any]]:
    matched_data = []
    total = min(len(predictions), len(originals))
    
    for idx in range(total):
        pred = predictions[idx]
        original = originals.iloc[idx]

        matched = {
            'id': idx, 
            'filename': pred.get('filename', ''),
            'c1_model': pred['c1'],
            'c2_model': pred['c2'],
            'c3_model': pred['c3'],
            'c4_model': pred['c4'],
            'c1_target': original['C1'],
            'c2_target': original['C2'],
            'c3_target': original['C3'],
            'c4_target': original['C4'],
            'elapsed': pred.get('elapsed', 0)
        }
        matched_data.append(matched)

    return matched_data

def root_mean_squared_error(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))

def calculate_metrics(matched_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    c1_pred = [d['c1_model'] for d in matched_data]
    c2_pred = [d['c2_model'] for d in matched_data]
    c3_pred = [d['c3_model'] for d in matched_data]
    c4_pred = [d['c4_model'] for d in matched_data]
    c1_true = [d['c1_target'] for d in matched_data]
    c2_true = [d['c2_target'] for d in matched_data]
    c3_true = [d['c3_target'] for d in matched_data]
    c4_true = [d['c4_target'] for d in matched_data]
    mse_c1 = mean_squared_error(c1_true, c1_pred)
    mse_c2 = mean_squared_error(c2_true, c2_pred)
    mse_c3 = mean_squared_error(c3_true, c3_pred)
    mse_c4 = mean_squared_error(c4_true, c4_pred)
    rmse_c1 = root_mean_squared_error(c1_true, c1_pred)
    rmse_c2 = root_mean_squared_error(c2_true, c2_pred)
    rmse_c3 = root_mean_squared_error(c3_true, c3_pred)
    rmse_c4 = root_mean_squared_error(c4_true, c4_pred)
    mae_c1 = mean_absolute_error(c1_true, c1_pred)
    mae_c2 = mean_absolute_error(c2_true, c2_pred)
    mae_c3 = mean_absolute_error(c3_true, c3_pred)
    mae_c4 = mean_absolute_error(c4_true, c4_pred)
    nmse_c1 = mse_c1 / (5**2)
    nmse_c2 = mse_c2 / (5**2)
    nmse_c3 = mse_c3 / (5**2)
    nmse_c4 = mse_c4 / (5**2)
    nrmse_c1 = rmse_c1 / 5
    nrmse_c2 = rmse_c2 / 5
    nrmse_c3 = rmse_c3 / 5
    nrmse_c4 = rmse_c4 / 5
    nmae_c1 = mae_c1 / 5
    nmae_c2 = mae_c2 / 5
    nmae_c3 = mae_c3 / 5
    nmae_c4 = mae_c4 / 5
    exact_c1 = sum(1 for t, p in zip(c1_true, c1_pred) if t == p) / len(c1_true) * 100
    exact_c2 = sum(1 for t, p in zip(c2_true, c2_pred) if t == p) / len(c2_true) * 100
    exact_c3 = sum(1 for t, p in zip(c3_true, c3_pred) if t == p) / len(c3_true) * 100
    exact_c4 = sum(1 for t, p in zip(c4_true, c4_pred) if t == p) / len(c4_true) * 100

    metrics = {
        'mse': {
            'c1': mse_c1,
            'c2': mse_c2,
            'c3': mse_c3,
            'c4': mse_c4,
        },
        'rmse': {
            'c1': rmse_c1,
            'c2': rmse_c2,
            'c3': rmse_c3,
            'c4': rmse_c4,
        },
        'normalized_mse': {
            'c1': nmse_c1,
            'c2': nmse_c2,
            'c3': nmse_c3,
            'c4': nmse_c4,
        },
        'normalized_rmse': {
            'c1': nrmse_c1,
            'c2': nrmse_c2,
            'c3': nrmse_c3,
            'c4': nrmse_c4,
        },
        'mae': {
            'c1': mae_c1,
            'c2': mae_c2,
            'c3': mae_c3,
            'c4': mae_c4,
        },
        'normalized_mae': {
            'c1': nmae_c1,
            'c2': nmae_c2,
            'c3': nmae_c3,
            'c4': nmae_c4,
        },
        'exact_match_percentage': {
            'c1': exact_c1,
            'c2': exact_c2,
            'c3': exact_c3,
            'c4': exact_c4,
        }
    }
    return metrics

def generate_error_ranking(metrics: Dict[str, Any]) -> List[Tuple[str, float]]:
    nrmse_values = metrics['normalized_rmse']
    competencies = ['c1', 'c2', 'c3', 'c4']
    error_ranking = [(comp, nrmse_values[comp]) for comp in competencies]
    error_ranking.sort(key=lambda x: x[1])
    return error_ranking

def save_results(matched_data: List[Dict[str, Any]], metrics: Dict[str, Any], output_dir: str, model_name: str = None):
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    prefix = f"{model_name}_" if model_name else ""
    df = pd.DataFrame(matched_data)
    csv_path = output_path / f'{prefix}essay_evaluation_results.csv'
    df.to_csv(csv_path, index=False)
    error_ranking = generate_error_ranking(metrics)
    ranking_dict = {
        'error_ranking': [{'competency': comp, 'rmse_normalized': error} for comp, error in error_ranking],
        'metrics': metrics
    }
    json_path = output_path / f'{prefix}evaluation_metrics.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(ranking_dict, f, indent=2)
    return csv_path, json_path

def print_summary(metrics: Dict[str, Any], error_ranking: List[Tuple[str, float]], num_samples: int):
    print("\n=== Essay Evaluation Summary ===")
    print(f"Number of samples evaluated: {num_samples}")
    print("\nMSE (Não normalizado, lower is better):")
    for comp, value in metrics['mse'].items():
        print(f"  {comp}: {value:.4f}")
    print("\nRMSE (Não normalizado, lower is better):")
    for comp, value in metrics['rmse'].items():
        print(f"  {comp}: {value:.4f}")
    print("\nNormalized MSE (0-1 scale, lower is better):")
    for comp, value in metrics['normalized_mse'].items():
        print(f"  {comp}: {value:.4f}")
    print("\nNormalized RMSE (0-1 scale, lower is better):")
    for comp, value in metrics['normalized_rmse'].items():
        print(f"  {comp}: {value:.4f}")
    print("\nMAE (lower is better, competencies in range 0-5 and total_score 0-1000):")
    for comp, value in metrics['mae'].items():
        print(f"  {comp}: {value:.4f}")
    print("\nNormalized MAE (0-1 scale, lower is better):")
    for comp, value in metrics['normalized_mae'].items():
        print(f"  {comp}: {value:.4f}")
    print("\nExact Match Percentage:")
    for comp, value in metrics['exact_match_percentage'].items():
        print(f"  {comp}: {value:.2f}%")
    print("\nCompetencies Ranked by Performance (from best to worst):")
    for i, (comp, error) in enumerate(error_ranking, 1):
        comp_name = {
            'c1': 'Competência 1 (formal language)',
            'c2': 'Competência 2 (understanding topic)',
            'c3': 'Competência 3 (argument organization)',
            'c4': 'Competência 4 (linguistic mechanisms)',
        }.get(comp, comp)
        print(f"  {i}. {comp_name}: {error:.4f} NRMSE")

@click.command()
@click.option('--csv', required=True, help='Path to the original CSV file with actual scores')
@click.option('--predictions-dir', required=True, help='Directory containing prediction JSON files')
@click.option('--output-dir', default='./evaluation_results', help='Directory to save evaluation results')
@click.option('--model', default=None, help='Optional filter for model names (e.g., "anthropic")')
def evaluate(csv: str, predictions_dir: str, output_dir: str, model: str):
    # Add some debugging info about paths
    print(f"Current working directory: {os.getcwd()}")
    print(f"CSV path: {os.path.abspath(csv)}")
    print(f"Predictions directory: {os.path.abspath(predictions_dir)}")
    
    original_data = load_original_data(csv)
    predictions = load_prediction_files(predictions_dir, model)
    print(f"Loaded {len(predictions)} prediction files")
    matched_data = match_predictions_with_originals(predictions, original_data)
    print(f"Successfully matched {len(matched_data)} predictions with original data")
    if not matched_data:
        print("No matched data found. Check your IDs or file structure.")
        return
    metrics = calculate_metrics(matched_data)
    error_ranking = generate_error_ranking(metrics)
    csv_path, json_path = save_results(matched_data, metrics, output_dir, model)
    print_summary(metrics, error_ranking, len(matched_data))
    print(f"\nDetailed results saved to {csv_path}")
    print(f"Metrics saved to {json_path}")

if __name__ == '__main__':
    evaluate()

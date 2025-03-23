import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import click
from sklearn.metrics import mean_squared_error
from typing import Dict, List, Any, Tuple

def load_original_data(csv_path: str) -> pd.DataFrame:
    """Load the original CSV file with actual essay scores."""
    df = pd.read_csv(csv_path)
    return df

def load_prediction_files(predictions_dir: str, model_filter: str = None) -> List[Dict[str, Any]]:
    """Load prediction JSON files from the given directory, optionally filtering by model name."""
    predictions_path = Path(predictions_dir)
    
    if model_filter:
        prediction_files = list(predictions_path.glob(f"{model_filter}*.json"))
    else:
        prediction_files = list(predictions_path.glob("*.json"))
    
    predictions = []
    for file in prediction_files:
        with open(file, 'r', encoding='utf-8') as f:
            try:
                prediction = json.load(f)
                # Add the filename to the prediction data
                prediction['filename'] = file.name
                predictions.append(prediction)
            except json.JSONDecodeError:
                print(f"Error loading JSON from {file}")
    
    return predictions

def match_predictions_with_originals(predictions: List[Dict[str, Any]], originals: pd.DataFrame) -> List[Dict[str, Any]]:
    """Match predictions with their original data by ID."""
    matched_data = []
    
    for pred in predictions:
        essay_id = pred.get('id')
        if essay_id is None:
            filename = pred.get('filename', '')
            try:
                essay_id = int(filename.split('_')[-1].split('.')[0])
            except (ValueError, IndexError):
                continue
        
        if essay_id < len(originals):
            original = originals.iloc[essay_id]
            
            matched = {
                'id': essay_id,
                'filename': pred.get('filename', ''),
                'c1_model': pred['c1'],
                'c2_model': pred['c2'],
                'c3_model': pred['c3'],
                'c4_model': pred['c4'],
                'c5_model': pred['c5'],
                'score_model': pred['total_score'],
                'c1_target': original['C1'],
                'c2_target': original['C2'],
                'c3_target': original['C3'],
                'c4_target': original['C4'],
                'c5_target': original['C5'],
                'score_target': original['score'],
                'elapsed': pred.get('elapsed', 0)
            }
            matched_data.append(matched)
    
    return matched_data

def calculate_metrics(matched_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate normalized MSE and RMSE for each competency and total score."""
    c1_pred = [d['c1_model'] for d in matched_data]
    c2_pred = [d['c2_model'] for d in matched_data]
    c3_pred = [d['c3_model'] for d in matched_data]
    c4_pred = [d['c4_model'] for d in matched_data]
    c5_pred = [d['c5_model'] for d in matched_data]
    score_pred = [d['score_model'] for d in matched_data]
    
    c1_true = [d['c1_target'] for d in matched_data]
    c2_true = [d['c2_target'] for d in matched_data]
    c3_true = [d['c3_target'] for d in matched_data]
    c4_true = [d['c4_target'] for d in matched_data]
    c5_true = [d['c5_target'] for d in matched_data]
    score_true = [d['score_target'] for d in matched_data]
    
    mse_c1 = mean_squared_error(c1_true, c1_pred)
    mse_c2 = mean_squared_error(c2_true, c2_pred)
    mse_c3 = mean_squared_error(c3_true, c3_pred)
    mse_c4 = mean_squared_error(c4_true, c4_pred)
    mse_c5 = mean_squared_error(c5_true, c5_pred)
    mse_score = mean_squared_error(score_true, score_pred)
    
    rmse_c1 = np.sqrt(mse_c1)
    rmse_c2 = np.sqrt(mse_c2)
    rmse_c3 = np.sqrt(mse_c3)
    rmse_c4 = np.sqrt(mse_c4)
    rmse_c5 = np.sqrt(mse_c5)
    rmse_score = np.sqrt(mse_score)
    
    nmse_c1 = mse_c1 / (200**2)
    nmse_c2 = mse_c2 / (200**2)
    nmse_c3 = mse_c3 / (200**2)
    nmse_c4 = mse_c4 / (200**2)
    nmse_c5 = mse_c5 / (200**2)
    nmse_score = mse_score / (1000**2)
    
    nrmse_c1 = rmse_c1 / 200
    nrmse_c2 = rmse_c2 / 200
    nrmse_c3 = rmse_c3 / 200
    nrmse_c4 = rmse_c4 / 200
    nrmse_c5 = rmse_c5 / 200
    nrmse_score = rmse_score / 1000
    
    exact_c1 = sum(1 for t, p in zip(c1_true, c1_pred) if t == p) / len(c1_true) * 100
    exact_c2 = sum(1 for t, p in zip(c2_true, c2_pred) if t == p) / len(c2_true) * 100
    exact_c3 = sum(1 for t, p in zip(c3_true, c3_pred) if t == p) / len(c3_true) * 100
    exact_c4 = sum(1 for t, p in zip(c4_true, c4_pred) if t == p) / len(c4_true) * 100
    exact_c5 = sum(1 for t, p in zip(c5_true, c5_pred) if t == p) / len(c5_true) * 100
    exact_score = sum(1 for t, p in zip(score_true, score_pred) if t == p) / len(score_true) * 100
    
    metrics = {
        'normalized_mse': {
            'c1': nmse_c1,
            'c2': nmse_c2,
            'c3': nmse_c3,
            'c4': nmse_c4,
            'c5': nmse_c5,
            'total_score': nmse_score
        },
        'normalized_rmse': {
            'c1': nrmse_c1,
            'c2': nrmse_c2,
            'c3': nrmse_c3,
            'c4': nrmse_c4,
            'c5': nrmse_c5,
            'total_score': nrmse_score
        },
        'exact_match_percentage': {
            'c1': exact_c1,
            'c2': exact_c2,
            'c3': exact_c3,
            'c4': exact_c4,
            'c5': exact_c5,
            'total_score': exact_score
        }
    }
    
    return metrics

def generate_error_ranking(metrics: Dict[str, Any]) -> List[Tuple[str, float]]:
    """Rank competencies by error (normalized RMSE), best-performing (lowest error) first."""
    nrmse_values = metrics['normalized_rmse']
    competencies = ['c1', 'c2', 'c3', 'c4', 'c5']
    
    error_ranking = [(comp, nrmse_values[comp]) for comp in competencies]
    
    error_ranking.sort(key=lambda x: x[1])
    
    return error_ranking

def save_results(matched_data: List[Dict[str, Any]], metrics: Dict[str, Any], output_dir: str, model_name: str = None):
    """Save the evaluation results to CSV and JSON files."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    prefix = f"{model_name}_" if model_name else ""
    
    df = pd.DataFrame(matched_data)
    csv_path = output_path / f'{prefix}essay_evaluation_results.csv'
    df.to_csv(csv_path, index=False)
    
    error_ranking = generate_error_ranking(metrics)
    ranking_dict = {
        'error_ranking': [{'competency': comp, 'rmse': error} for comp, error in error_ranking],
        'metrics': metrics
    }
    
    json_path = output_path / f'{prefix}evaluation_metrics.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(ranking_dict, f, indent=2)
    
    return df, ranking_dict, csv_path, json_path

def print_summary(metrics: Dict[str, Any], error_ranking: List[Tuple[str, float]], num_samples: int):
    """Print a summary of the evaluation results."""
    print("\n=== Essay Evaluation Summary ===")
    print(f"Number of samples evaluated: {num_samples}")
    
    print("\nNormalized MSE (0-1 scale, lower is better):")
    for comp, value in metrics['normalized_mse'].items():
        print(f"  {comp}: {value:.4f}")
    
    print("\nNormalized RMSE (0-1 scale, lower is better):")
    for comp, value in metrics['normalized_rmse'].items():
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
            'c5': 'Competência 5 (solution proposal)'
        }.get(comp, comp)
        print(f"  {i}. {comp_name}: {error:.4f} NRMSE")

@click.command()
@click.option('--csv', required=True, help='Path to the original CSV file with actual scores')
@click.option('--predictions-dir', required=True, help='Directory containing prediction JSON files')
@click.option('--output-dir', default='./evaluation_results', help='Directory to save evaluation results')
@click.option('--model', default=None, help='Optional filter for model names (e.g., "anthropic")')
def evaluate(original_csv: str, predictions_dir: str, output_dir: str, model_filter: str):
    """Evaluate essay score predictions against actual scores."""
    original_data = load_original_data(original_csv)
    predictions = load_prediction_files(predictions_dir, model_filter)
    
    print(f"Loaded {len(predictions)} prediction files")
    
    matched_data = match_predictions_with_originals(predictions, original_data)
    
    print(f"Successfully matched {len(matched_data)} predictions with original data")
    
    if not matched_data:
        print("No matched data found. Check your IDs or file structure.")
        return
    
    metrics = calculate_metrics(matched_data)
    error_ranking = generate_error_ranking(metrics)
    
    csv_path, json_path = save_results(matched_data, metrics, output_dir, model_filter)
    
    print_summary(metrics, error_ranking, len(matched_data))
    
    print(f"\nDetailed results saved to {csv_path}")
    print(f"Metrics saved to {json_path}")

if __name__ == '__main__':
    evaluate()

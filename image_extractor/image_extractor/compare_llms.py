"""
Script to compare LLM detection results against ground truth and generate metrics.
Generates a CSV summary for each model (openai, anthropic, gemini) and prints a final report.
"""
import os
import json
import csv
from statistics import mean

def compute_metrics(y_true, y_pred):
    classes = sorted(set(y_true) | {yp for yp in y_pred if yp is not None})
    tp = {c: 0 for c in classes}
    fp = {c: 0 for c in classes}
    fn = {c: 0 for c in classes}
    for yt, yp in zip(y_true, y_pred):
        if yp is None:
            fn[yt] += 1
        elif yt == yp:
            tp[yt] += 1
        else:
            fp[yp] += 1
            fn[yt] += 1
    precision_c = {}
    recall_c = {}
    for c in classes:
        precision_c[c] = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
        recall_c[c]    = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0
    precision = mean(precision_c.values())
    recall    = mean(recall_c.values())
    total = len(y_true)
    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    accuracy = correct / total if total > 0 else 0.0
    error_rate = 1.0 - accuracy
    most_missed = max(fn.items(), key=lambda x: x[1])[0] if fn else None
    return {
        'total': total,
        'correct': correct,
        'incorrect': total - correct,
        'accuracy': accuracy,
        'error_rate': error_rate,
        'precision': precision,
        'recall': recall,
        'most_missed': most_missed
    }

def are_labels_equivalent(label1, label2):
    """Checks if two labels are equivalent, treating 'invalid' and 'invalido' the same."""
    if label1 == label2:
        return True
    # Consider None explicitly if necessary, but direct comparison handles it if both are None.
    # If one is None and the other is not, they are not equivalent unless the non-None is handled below.
    if label1 is None or label2 is None:
        return False # None is not equivalent to 'invalid' or 'invalido'
    return (label1 == 'invalid' and label2 == 'invalido') or \
           (label1 == 'invalido' and label2 == 'invalid')

def process_model(base_dir, model):
    results_dir = os.path.join(base_dir, 'results_llms', model)
    gt_dir = os.path.join(base_dir, 'json_gt', 'test')
    y_true_normalized = [] # For metrics calculation
    y_pred_normalized = [] # For metrics calculation
    csv_rows = []
    gt_files = [f for f in os.listdir(gt_dir) if f.endswith('.json')]
    for fname in os.listdir(results_dir):
        if not fname.endswith('.json'):
            continue
        matches = [g for g in gt_files if fname.endswith(g)]
        if not matches:
            print(f"Warning: GT file not found for result '{fname}'")
            continue
        gt_fname = matches[0]
        res_path = os.path.join(results_dir, fname)
        gt_path = os.path.join(gt_dir, gt_fname)
        try:
            with open(res_path, 'r', encoding='utf-8') as f:
                res = json.load(f)
            with open(gt_path, 'r', encoding='utf-8') as f:
                gt = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON in {fname} or {gt_fname}: {e}")
            continue
        except FileNotFoundError as e:
            print(f"Error opening file: {e}")
            continue

        # Ensure 'detections' key exists
        gt_detections = gt.get('detections', {})
        res_detections = res.get('detections', {})

        gt_map = {k: v.get('name') for k, v in gt_detections.items() if v.get('name') is not None}
        # Use gt_map keys to ensure we only compare items present in ground truth
        pred_map = {k: res_detections.get(k, {}).get('name') for k in gt_map}

        file_total = len(gt_map)
        file_correct = 0
        current_file_correct_keys = []
        current_file_error_details = []

        for k, true_label in gt_map.items():
            pred_label = pred_map.get(k) # Can be None if key k missing in res or name missing

            # Normalize 'invalido' to 'invalid' for metrics calculation
            norm_true = 'invalid' if true_label == 'invalido' else true_label
            norm_pred = 'invalid' if pred_label == 'invalido' else pred_label
            y_true_normalized.append(norm_true)
            y_pred_normalized.append(norm_pred) # Appends None if pred_label was None

            # Use equivalence check for file accuracy and CSV reporting (using original labels)
            if are_labels_equivalent(true_label, pred_label):
                file_correct += 1
                current_file_correct_keys.append(k)
            else:
                pred_repr = pred_label if pred_label is not None else '' # Represent None as empty string in error json
                current_file_error_details.append(f"{k}:{pred_repr}")

        file_accuracy = file_correct / file_total if file_total else 0.0
        csv_rows.append({
            'file': os.path.relpath(res_path, base_dir),
            'json_model': json.dumps(pred_map, ensure_ascii=False), # Original predictions
            'json_gt': json.dumps(gt_map, ensure_ascii=False),      # Original ground truth
            'correct_questions': json.dumps(current_file_correct_keys),
            'error_questions': json.dumps(current_file_error_details, ensure_ascii=False),
            'file_accuracy': str(round(file_accuracy, 4)).replace('.', ',')
        })

    # Pass normalized lists to compute_metrics
    metrics = compute_metrics(y_true_normalized, y_pred_normalized)
    csv_path = os.path.join(base_dir, f'results_{model}.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['file', 'json_model', 'json_gt', 'correct_questions', 'error_questions', 'file_accuracy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    return metrics

if __name__ == '__main__':
    base = os.path.dirname(os.path.abspath(__file__))
    models = ['openai', 'anthropic', 'google', 'json_image_yolo'] # Added 'json_image_yolo'
    available = []
    for m in models:
        path = os.path.join(base, 'results_llms', m)
        if os.path.isdir(path):
            available.append(m)
        else:
            print(f"Warning: results folder for model '{m}' not found at {path}, skipping.")
    all_results = {}
    for m in available:
        print(f"Processing model: {m}")
        all_results[m] = process_model(base, m)
    print("\nReport:")
    for m, met in all_results.items():
        print(f"Model: {m}")
        print(f"  Total: {met['total']}")
        print(f"  Correct: {met['correct']}  Incorrect: {met['incorrect']}")
        print(f"  Accuracy: {met['accuracy']:.2%}  Error rate: {met['error_rate']:.2%}")
        print(f"  Precision: {met['precision']:.2%}  Recall: {met['recall']:.2%}")
        print(f"  Most missed letter: {met['most_missed']}\n")
    # Generate the list of expected CSV files dynamically
    generated_csvs = [f'results_{m}.csv' for m in available]
    print(f"CSV files generated: {', '.join(generated_csvs)}")

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

def process_model(base_dir, model):
    results_dir = os.path.join(base_dir, 'results_llms', model)
    gt_dir = os.path.join(base_dir, 'json_gt', 'test')
    y_true = []
    y_pred = []
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
        with open(res_path, 'r', encoding='utf-8') as f:
            res = json.load(f)
        with open(gt_path, 'r', encoding='utf-8') as f:
            gt = json.load(f)
        gt_map = {k: v['name'] for k, v in gt['detections'].items()}
        pred_map = {k: res['detections'].get(k, {}).get('name', '') for k in gt_map}
        file_total = len(gt_map)
        file_correct = 0
        for k, true_label in gt_map.items():
            pred_label = pred_map.get(k) or None
            y_true.append(true_label)
            y_pred.append(pred_label)
            if pred_label == true_label:
                file_correct += 1
        file_accuracy = file_correct / file_total if file_total else 0.0
        csv_rows.append({
            'file': os.path.relpath(res_path, base_dir),
            'json_model': json.dumps(pred_map, ensure_ascii=False),
            'json_gt': json.dumps(gt_map, ensure_ascii=False),
            'correct_questions': json.dumps([k for k in gt_map if pred_map.get(k) == gt_map[k]]),
            'error_questions': json.dumps([f"{k}:{pred_map.get(k, '')}" for k in gt_map if pred_map.get(k) != gt_map[k]], ensure_ascii=False),
            'file_accuracy': str(round(file_accuracy, 4)).replace('.', ',')
        })
    metrics = compute_metrics(y_true, y_pred)
    csv_path = os.path.join(base_dir, f'results_{model}.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['file', 'json_model', 'json_gt', 'correct_questions', 'error_questions', 'file_accuracy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    return metrics

if __name__ == '__main__':
    base = os.path.dirname(os.path.abspath(__file__))
    models = ['openai', 'anthropic', 'google']
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
    print("CSV files generated: results_openai.csv, results_anthropic.csv, results_google.csv")

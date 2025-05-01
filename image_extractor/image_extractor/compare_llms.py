#!/usr/bin/env python3
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
        for key, item in gt['detections'].items():
            true_label = item['name']
            pred_label = res['detections'].get(key, {}).get('name')
            if pred_label is None:
                pred_label = None
            y_true.append(true_label)
            y_pred.append(pred_label)
    metrics = compute_metrics(y_true, y_pred)
    csv_path = os.path.join(base_dir, f'results_{model}.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)
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

#!/usr/bin/env python3
import os
import json
import re
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple, Any
from dotenv import load_dotenv
from sklearn.metrics import precision_score, recall_score, f1_score

load_dotenv()

def load_model_results(results_folder: str, model: str) -> Dict[str, Dict]:
    """
    Carrega os resultados das extrações matemáticas dos arquivos JSON
    """
    results = {}
    path = Path(results_folder)
    
    if not path.exists():
        print(f"Aviso: Diretório de resultados não encontrado em {results_folder}")
        return results
    
    # Determina o prefixo do modelo
    if model == "openai":
        model_prefix = "openai_"
    elif model == "google":
        model_prefix = "google_"
    elif model == "anthropic":
        model_prefix = "anthropic_"
    else:
        model_prefix = f"{model}_"
        
    # Busca por todos arquivos JSON do modelo específico
    json_files = list(path.glob(f"**/{model_prefix}*_math.json"))
    
    if not json_files:
        print(f"Aviso: Nenhum arquivo JSON encontrado com o padrão {model_prefix}*_math.json em {results_folder}")
        return results
    
    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                
                # Extrai o número da imagem do nome do arquivo
                filename = json_file.name
                match = re.search(r'_(\d+)_math\.json$', filename)
                if match:
                    image_num = int(match.group(1))
                    
                    # Tenta diferentes campos possíveis para a expressão matemática
                    expression = None
                    for field in ['expression', 'main_text', 'text', 'formula', 'result']:
                        if field in data and data[field]:
                            expression = data[field]
                            break
                    
                    if not expression:
                        continue
                        
                    # Extrai a categoria (horizontais, verticais, ruins)
                    category_match = re.search(r'([^_]+)_', filename)
                    category = category_match.group(1) if category_match else "unknown"
                    
                    # Armazena o resultado
                    results[image_num] = {
                        "expression": str(expression).replace('"', '').strip(),
                        "category": category,
                        "path": filename,
                        "elapsed": data.get("elapsed", 0.0)
                    }
        except Exception as e:
            print(f"Erro ao processar o arquivo {json_file}: {e}")
    
    return results


def load_ground_truth(labels_file: str) -> Dict[int, str]:
    """
    Carrega o ground truth do arquivo de labels
    """
    ground_truth = {}
    try:
        with open(labels_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    # Extrai o número da imagem (ex: "4.png" -> 4)
                    image_name = parts[0]
                    image_num = int(image_name.split('.')[0])
                    expression = parts[1].strip()
                    ground_truth[image_num] = expression
    except Exception as e:
        print(f"Erro ao carregar o arquivo de ground truth: {e}")
    
    return ground_truth


def normalize_expression(expr: str) -> str:
    """
    Normaliza expressões matemáticas para facilitar a comparação
    """
    # Remove espaços entre dígitos
    normalized = re.sub(r'(\d)\s+(\d)', r'\1\2', expr)
    # Substitui múltiplos espaços por um único espaço
    normalized = re.sub(r'\s+', ' ', normalized)
    # Remove espaços antes e depois
    normalized = normalized.strip()
    # Substitui "=" por " = " para garantir espaçamento consistente
    normalized = normalized.replace('=', ' = ')
    # Normaliza operadores
    normalized = normalized.replace('+', ' + ').replace('-', ' - ').replace('*', ' * ').replace('/', ' / ')
    normalized = normalized.replace('×', ' * ').replace('÷', ' / ')
    # Remove espaços duplicados resultantes
    normalized = re.sub(r'\s+', ' ', normalized)
    
    return normalized


def identify_operation(expr: str) -> str:
    """
    Identifica a operação matemática na expressão
    """
    if '+' in expr:
        return 'addition'
    elif '-' in expr:
        return 'subtraction'
    elif '*' in expr or 'x' in expr or '×' in expr:
        return 'multiplication'
    elif '/' in expr or '÷' in expr:
        return 'division'
    else:
        # Se não houver operador, assume que é apenas um número
        return 'number'


def extract_digits(text: str) -> str:
    """
    Extrai apenas dígitos de um texto
    """
    return ''.join(filter(str.isdigit, text))


def evaluate_expression(pred_expr: str, true_expr: str) -> Dict[str, Any]:
    """
    Avalia a precisão da extração da expressão matemática
    """
    pred_normalized = normalize_expression(pred_expr)
    true_normalized = normalize_expression(true_expr)
    
    # Verifica igualdade exata após normalização
    exact_match = pred_normalized == true_normalized
    
    # Identifica diferenças entre as expressões
    difference = ""
    if not exact_match:
        if pred_normalized != true_normalized:
            difference = f"Expressão diferente: '{true_normalized}' vs '{pred_normalized}'"
    
    # Identifica a operação baseada no ground truth
    operation = identify_operation(true_normalized)
    
    # Tenta extrair os números e o resultado
    numbers_pred = []
    numbers_true = []
    result_pred = None
    result_true = None
    
    # Extrai números da expressão verdadeira
    if '=' in true_normalized:
        parts = true_normalized.split('=')
        left_side = parts[0].strip()
        right_side = parts[1].strip() if len(parts) > 1 else ""
        
        # Extrai números da expressão (lado esquerdo)
        if '+' in left_side:
            numbers_true = [extract_digits(num.strip()) for num in left_side.split('+')]
            numbers_true = [int(num) for num in numbers_true if num]
        elif '-' in left_side:
            numbers_true = [extract_digits(num.strip()) for num in left_side.split('-')]
            numbers_true = [int(num) for num in numbers_true if num]
        elif '*' in left_side or 'x' in left_side or '×' in left_side:
            numbers_true = [extract_digits(num.strip()) for num in re.split(r'[*x×]', left_side)]
            numbers_true = [int(num) for num in numbers_true if num]
        elif '/' in left_side or '÷' in left_side:
            numbers_true = [extract_digits(num.strip()) for num in re.split(r'[/÷]', left_side)]
            numbers_true = [int(num) for num in numbers_true if num]
            
        # Extrai o resultado (lado direito)
        result_digits = extract_digits(right_side)
        if result_digits:
            result_true = int(result_digits)
    else:
        # Se não houver '=', tente extrair os números diretamente
        if '+' in true_normalized:
            numbers_true = [extract_digits(num.strip()) for num in true_normalized.split('+')]
            numbers_true = [int(num) for num in numbers_true if num]
        elif '-' in true_normalized:
            numbers_true = [extract_digits(num.strip()) for num in true_normalized.split('-')]
            numbers_true = [int(num) for num in numbers_true if num]
        elif '*' in true_normalized or 'x' in true_normalized or '×' in true_normalized:
            numbers_true = [extract_digits(num.strip()) for num in re.split(r'[*x×]', true_normalized)]
            numbers_true = [int(num) for num in numbers_true if num]
        elif '/' in true_normalized or '÷' in true_normalized:
            numbers_true = [extract_digits(num.strip()) for num in re.split(r'[/÷]', true_normalized)]
            numbers_true = [int(num) for num in numbers_true if num]
    
    # Extrai números da expressão predita
    if '=' in pred_normalized:
        parts = pred_normalized.split('=')
        left_side = parts[0].strip()
        right_side = parts[1].strip() if len(parts) > 1 else ""
        
        # Extrai números da expressão (lado esquerdo)
        if '+' in left_side:
            numbers_pred = [extract_digits(num.strip()) for num in left_side.split('+')]
            numbers_pred = [int(num) for num in numbers_pred if num]
        elif '-' in left_side:
            numbers_pred = [extract_digits(num.strip()) for num in left_side.split('-')]
            numbers_pred = [int(num) for num in numbers_pred if num]
        elif '*' in left_side or 'x' in left_side or '×' in left_side:
            numbers_pred = [extract_digits(num.strip()) for num in re.split(r'[*x×]', left_side)]
            numbers_pred = [int(num) for num in numbers_pred if num]
        elif '/' in left_side or '÷' in left_side:
            numbers_pred = [extract_digits(num.strip()) for num in re.split(r'[/÷]', left_side)]
            numbers_pred = [int(num) for num in numbers_pred if num]
            
        # Extrai o resultado (lado direito)
        result_digits = extract_digits(right_side)
        if result_digits:
            result_pred = int(result_digits)
    else:
        # Se não houver '=', tente extrair os números diretamente
        if '+' in pred_normalized:
            numbers_pred = [extract_digits(num.strip()) for num in pred_normalized.split('+')]
            numbers_pred = [int(num) for num in numbers_pred if num]
        elif '-' in pred_normalized:
            numbers_pred = [extract_digits(num.strip()) for num in pred_normalized.split('-')]
            numbers_pred = [int(num) for num in numbers_pred if num]
        elif '*' in pred_normalized or 'x' in pred_normalized or '×' in pred_normalized:
            numbers_pred = [extract_digits(num.strip()) for num in re.split(r'[*x×]', pred_normalized)]
            numbers_pred = [int(num) for num in numbers_pred if num]
        elif '/' in pred_normalized or '÷' in pred_normalized:
            numbers_pred = [extract_digits(num.strip()) for num in re.split(r'[/÷]', pred_normalized)]
            numbers_pred = [int(num) for num in numbers_pred if num]
    
    # Verifica correspondência de números e resultado
    numbers_match = numbers_pred == numbers_true
    result_match = result_pred == result_true
    
    # Verifica correspondência parcial (números corretos mas ordem errada)
    numbers_partial_match = sorted(numbers_pred) == sorted(numbers_true) if numbers_pred and numbers_true else False
    
    # Determina o nível de confiança (simulado para este contexto)
    if exact_match:
        confidence = 1.0
    elif numbers_match and result_match:
        confidence = 0.9
    elif numbers_match or result_match:
        confidence = 0.7
    elif numbers_partial_match:
        confidence = 0.5
    else:
        confidence = 0.2
        
    # Define o nível de confiança categorizado
    if confidence >= 0.8:
        confidence_level = "high"
    elif confidence >= 0.5:
        confidence_level = "medium"
    else:
        confidence_level = "low"
    
    # Atualiza a descrição da diferença com informações mais detalhadas
    if not exact_match:
        if not difference:
            difference = "Expressão diferente"
            
        if not numbers_match and numbers_pred and numbers_true:
            difference += f"; Números diferentes: {numbers_true} vs {numbers_pred}"
            
        if not result_match and result_true is not None and result_pred is not None:
            difference += f"; Resultado diferente: {result_true} vs {result_pred}"
    
    return {
        'exact_match': exact_match,
        'numbers_match': numbers_match,
        'result_match': result_match,
        'numbers_partial_match': numbers_partial_match,
        'operation': operation,
        'confidence': confidence,
        'confidence_level': confidence_level,
        'difference': difference
    }


def calculate_metrics(extracted_results: Dict[int, Dict], 
                      ground_truth: Dict[int, str]) -> Tuple[pd.DataFrame, Dict]:
    """
    Calcula métricas de avaliação para extração de expressões matemáticas, 
    seguindo a abordagem do answer_sheet_eval.py
    """
    eval_data = []
    summary = {
        "total_images": len(ground_truth),
        "processed_images": 0,
        "correct_expressions": 0,
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1_score": 0.0,
        "avg_confidence": 0.0,
        "total_time": 0.0,
        "expressions_by_confidence": {
            "high": {"total": 0, "correct": 0, "accuracy": 0.0},
            "medium": {"total": 0, "correct": 0, "accuracy": 0.0},
            "low": {"total": 0, "correct": 0, "accuracy": 0.0},
        },
        "expressions_by_operation": {},
        "expressions_by_category": {}
    }
    
    # Inicializa contadores por operação e categoria
    operation_types = ["addition", "subtraction", "multiplication", "division", "number"]
    category_types = ["horizontais", "verticais", "ruins"]
    
    for op in operation_types:
        summary["expressions_by_operation"][op] = {
            "total_ground_truth": 0,
            "processed": 0,
            "correct": 0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "numbers_match": 0,
            "result_match": 0,
            "partial_match": 0
        }
    
    for cat in category_types:
        summary["expressions_by_category"][cat] = {
            "total_ground_truth": 0,
            "processed": 0,
            "correct": 0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0
        }
    
    # Primeiro, contar operações no ground truth
    for img_num, true_expr in ground_truth.items():
        operation = identify_operation(true_expr)
        summary["expressions_by_operation"][operation]["total_ground_truth"] += 1
        
        # Determina a categoria a partir do resultado extraído (se disponível) ou assume "unknown"
        category = extracted_results.get(img_num, {}).get("category", "unknown")
        if category in category_types:
            summary["expressions_by_category"][category]["total_ground_truth"] += 1
    
    # Listas para calcular precisão, recall e F1
    y_true = []
    y_pred = []
    
    # Dicionários para métricas por operação e categoria
    y_true_by_op = {op: [] for op in operation_types}
    y_pred_by_op = {op: [] for op in operation_types}
    y_true_by_cat = {cat: [] for cat in category_types}
    y_pred_by_cat = {cat: [] for cat in category_types}
    
    # Avalia cada expressão
    for img_num, true_expr in ground_truth.items():
        if img_num in extracted_results:
            extracted_data = extracted_results[img_num]
            pred_expr = extracted_data["expression"]
            category = extracted_data["category"]
            
            summary["processed_images"] += 1
            summary["total_time"] += extracted_data.get("elapsed", 0)
            
            # Avalia a extração
            evaluation = evaluate_expression(pred_expr, true_expr)
            operation = evaluation["operation"]
            confidence = evaluation["confidence"]
            confidence_level = evaluation["confidence_level"]
            is_correct = evaluation["exact_match"]
            
            # Para cálculo de precisão/recall/F1
            y_true.append(1)  # Ground truth é sempre 1 (positivo)
            y_pred.append(1 if is_correct else 0)
            
            # Métricas por operação
            y_true_by_op[operation].append(1)
            y_pred_by_op[operation].append(1 if is_correct else 0)
            
            # Métricas por categoria
            if category in category_types:
                y_true_by_cat[category].append(1)
                y_pred_by_cat[category].append(1 if is_correct else 0)
            
            # Atualiza contadores por operação
            summary["expressions_by_operation"][operation]["processed"] += 1
            
            # Atualiza contadores por categoria
            if category in category_types:
                summary["expressions_by_category"][category]["processed"] += 1
            
            # Atualiza estatísticas de confiança
            summary["expressions_by_confidence"][confidence_level]["total"] += 1
            summary["avg_confidence"] += confidence
            
            # Registra acertos e contadores específicos
            if is_correct:
                summary["correct_expressions"] += 1
                summary["expressions_by_confidence"][confidence_level]["correct"] += 1
                summary["expressions_by_operation"][operation]["correct"] += 1
                if category in category_types:
                    summary["expressions_by_category"][category]["correct"] += 1
            
            # Registra matches parciais por operação
            if evaluation["numbers_match"]:
                summary["expressions_by_operation"][operation]["numbers_match"] += 1
            if evaluation["result_match"]:
                summary["expressions_by_operation"][operation]["result_match"] += 1
            if evaluation["numbers_partial_match"]:
                summary["expressions_by_operation"][operation]["partial_match"] += 1
                
            # Adiciona avaliação aos dados detalhados
            eval_data.append({
                "image_number": img_num,
                "ground_truth": true_expr,
                "result_llm": pred_expr,
                "is_correct": is_correct,
                "confidence": confidence,
                "operation": operation,
                "difference": evaluation.get("difference", "")
            })
        else:
            # Expressão não foi processada
            operation = identify_operation(true_expr)
            eval_data.append({
                "image_number": img_num,
                "ground_truth": true_expr,
                "result_llm": "missing",
                "is_correct": False,
                "confidence": 0.0,
                "operation": operation,
                "difference": "Expressão não processada"
            })
    
    # Calcula estatísticas resumidas
    if summary["processed_images"] > 0:
        summary["accuracy"] = summary["correct_expressions"] / summary["processed_images"]
        summary["avg_confidence"] /= summary["processed_images"]
        
        # Calcula precisão, recall e F1-score geral
        if len(y_pred) > 0:
            try:
                summary["precision"] = precision_score(y_true, y_pred, zero_division=0)
                summary["recall"] = recall_score(y_true, y_pred, zero_division=0)
                summary["f1_score"] = f1_score(y_true, y_pred, zero_division=0)
            except Exception as e:
                print(f"Erro ao calcular métricas: {e}")
                summary["precision"] = summary["recall"] = summary["f1_score"] = 0.0
        
        # Calcula acurácia por nível de confiança
        for level in ["high", "medium", "low"]:
            level_total = summary["expressions_by_confidence"][level]["total"]
            level_correct = summary["expressions_by_confidence"][level]["correct"]
            if level_total > 0:
                summary["expressions_by_confidence"][level]["accuracy"] = level_correct / level_total
        
        # Calcula métricas por operação
        for op in operation_types:
            op_data = summary["expressions_by_operation"][op]
            if op_data["processed"] > 0:
                op_data["accuracy"] = op_data["correct"] / op_data["processed"]
                
                # Calcula precisão, recall e F1-score por operação
                if len(y_pred_by_op[op]) > 0:
                    try:
                        op_data["precision"] = precision_score(y_true_by_op[op], y_pred_by_op[op], zero_division=0)
                        op_data["recall"] = recall_score(y_true_by_op[op], y_pred_by_op[op], zero_division=0)
                        op_data["f1_score"] = f1_score(y_true_by_op[op], y_pred_by_op[op], zero_division=0)
                    except Exception as e:
                        print(f"Erro ao calcular métricas para operação {op}: {e}")
                        op_data["precision"] = op_data["recall"] = op_data["f1_score"] = 0.0
            
        # Calcula métricas por categoria
        for cat in category_types:
            cat_data = summary["expressions_by_category"][cat]
            if cat_data["processed"] > 0:
                cat_data["accuracy"] = cat_data["correct"] / cat_data["processed"]
                
                # Calcula precisão, recall e F1-score por categoria
                if len(y_pred_by_cat[cat]) > 0:
                    try:
                        cat_data["precision"] = precision_score(y_true_by_cat[cat], y_pred_by_cat[cat], zero_division=0)
                        cat_data["recall"] = recall_score(y_true_by_cat[cat], y_pred_by_cat[cat], zero_division=0)
                        cat_data["f1_score"] = f1_score(y_true_by_cat[cat], y_pred_by_cat[cat], zero_division=0)
                    except Exception as e:
                        print(f"Erro ao calcular métricas para categoria {cat}: {e}")
                        cat_data["precision"] = cat_data["recall"] = cat_data["f1_score"] = 0.0
                
    return pd.DataFrame(eval_data), summary


def generate_report(eval_df: pd.DataFrame, summary: Dict, output_path: str, model_name: str):
    """
    Gera e salva o relatório de avaliação
    """
    # Seleciona apenas as colunas desejadas para o CSV
    columns_to_save = ["image_number", "ground_truth", "result_llm", "is_correct", 
                      "confidence", "operation", "difference"]
    
    # Verifica se todas as colunas existem
    for col in columns_to_save:
        if col not in eval_df.columns:
            print(f"Aviso: Coluna {col} não encontrada no DataFrame")
    
    # Salva resultados detalhados em CSV
    eval_df.to_csv(output_path, index=False, columns=[c for c in columns_to_save if c in eval_df.columns])
    
    # Salva sumário em JSON
    summary_path = output_path.replace(".csv", "_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    
    # Imprime relatório resumido
    print("\n===== AVALIAÇÃO DE EXTRAÇÃO DE EXPRESSÕES MATEMÁTICAS =====")
    print(f"Modelo: {model_name}")
    print(f"Total de imagens: {summary['total_images']}")
    print(f"Imagens processadas: {summary['processed_images']}")
    print(f"Expressões corretas: {summary['correct_expressions']}")
    print(f"Acurácia geral: {summary['accuracy']:.2%}")
    print(f"Precisão: {summary['precision']:.2%}")
    print(f"Recall: {summary['recall']:.2%}")
    print(f"F1-Score: {summary['f1_score']:.2%}")
    print(f"Confiança média: {summary['avg_confidence']:.2f}")
    print(f"Tempo total de processamento: {summary['total_time']:.2f} segundos")
    
    print("\nAcurácia por nível de confiança:")
    for level in ["high", "medium", "low"]:
        level_data = summary["expressions_by_confidence"][level]
        level_total = level_data["total"]
        if level_total > 0:
            print(f"  {level.title()} confiança ({level_total} expressões): {level_data['accuracy']:.2%} acurácia")
    
    print("\nDesempenho por operação:")
    for op, op_data in summary["expressions_by_operation"].items():
        if op_data["processed"] > 0:
            print(f"  {op.title()} ({op_data['processed']} de {op_data['total_ground_truth']}):")
            print(f"    Acurácia: {op_data['accuracy']:.2%}")
            print(f"    Precisão: {op_data['precision']:.2%}")
            print(f"    Recall: {op_data['recall']:.2%}")
            print(f"    F1-Score: {op_data['f1_score']:.2%}")
            
    print("\nDesempenho por categoria:")
    for cat, cat_data in summary["expressions_by_category"].items():
        if cat_data["processed"] > 0:
            print(f"  {cat.title()} ({cat_data['processed']} de {cat_data['total_ground_truth']}):")
            print(f"    Acurácia: {cat_data['accuracy']:.2%}")
            print(f"    Precisão: {cat_data['precision']:.2%}")
            print(f"    Recall: {cat_data['recall']:.2%}")
            print(f"    F1-Score: {cat_data['f1_score']:.2%}")
    
    print("\nMatriz de confusão por operação:")
    confusion_matrix = pd.crosstab(
        eval_df["operation"], 
        eval_df["is_correct"],
        rownames=["Operação"], 
        colnames=["Correto"],
        margins=True
    )
    print(confusion_matrix)


def main():
    parser = argparse.ArgumentParser(
        description="Avaliação de extração de expressões matemáticas por modelos LLM"
    )
    parser.add_argument(
        "--model", 
        required=True,
        choices=["openai", "google", "anthropic"], 
        help="Modelo utilizado na extração"
    )
    parser.add_argument(
        "--results-dir", 
        default="./", 
        help="Diretório contendo os arquivos de resultados"
    )
    parser.add_argument(
        "--labels-file",
        default="matching_labels.txt",
        help="Arquivo contendo o ground truth"
    )
    parser.add_argument(
        "--output-dir",
        default="./",
        help="Diretório para salvar os resultados da avaliação"
    )
    
    args = parser.parse_args()
    
    results_dir = args.results_dir
    labels_file = os.path.join(results_dir, args.labels_file)
    output_dir = args.output_dir
    model_name = args.model
    
    # Cria diretório de saída se não existir
    os.makedirs(output_dir, exist_ok=True)
    
    output_csv = os.path.join(output_dir, f"math_eval_{model_name}.csv")
    
    # Carrega dados
    extracted_results = load_model_results(results_dir, model_name)
    if not extracted_results:
        print(f"Nenhum resultado de extração encontrado para o modelo {model_name}.")
        return
        
    ground_truth = load_ground_truth(labels_file)
    if not ground_truth:
        print(f"Nenhum dado de ground truth encontrado em {labels_file}.")
        return
        
    # Calcula métricas e gera relatório
    eval_df, summary = calculate_metrics(extracted_results, ground_truth)
    generate_report(eval_df, summary, output_csv, model_name)
    
    print(f"\nAvaliação detalhada salva em {output_csv}")
    print(f"Sumário da avaliação salvo em {output_csv.replace('.csv', '_summary.json')}")


if __name__ == "__main__":
    main() 
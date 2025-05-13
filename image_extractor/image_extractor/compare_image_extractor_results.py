#!/usr/bin/env python3
import os
import json
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from sklearn.metrics import precision_score, recall_score, f1_score

def load_ground_truth(labels_file):
    """
    Carrega o ground truth do arquivo labels.txt
    """
    ground_truth = {}
    with open(labels_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                # Extrai o número da imagem (ex: "4.png" -> 4)
                image_name = parts[0]
                image_num = int(image_name.split('.')[0])
                expression = parts[1].strip()
                ground_truth[image_num] = expression
    
    return ground_truth

def load_llm_results(results_folder):
    """
    Carrega os resultados das LLMs dos arquivos JSON
    """
    results = {}
    processed_files = []
    for filename in os.listdir(results_folder):
        if filename.endswith('.json'):
            # Extrai a categoria, modelo LLM e número da imagem do nome do arquivo
            # Exemplo: horizontais_anthropic_3-7-sonnet-latest_4_math.json
            # Ou: horizontais_google__4_math.json (com duplo underscore)
            match = re.search(r'([^_]+)_([^_]+)(?:__|\d*_[^_]*)*_(\d+)_math\.json', filename)
            if match:
                category, llm, image_num = match.groups()
                image_num = int(image_num)
                processed_files.append(filename)
                
                # Normaliza os nomes das LLMs
                if 'anthropic' in llm or 'claude' in llm:
                    llm = 'anthropic'
                elif 'openai' in llm or 'gpt' in llm:
                    llm = 'openai'
                elif 'google' in llm or 'gemini' in llm:
                    llm = 'google'
                
                file_path = os.path.join(results_folder, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        try:
                            data = json.load(f)
                            
                            # Tenta diferentes campos possíveis para a expressão matemática
                            expression = None
                            for field in ['expression', 'main_text', 'text', 'formula', 'result']:
                                if field in data and data[field]:
                                    expression = data[field]
                                    break
                            
                            if not expression:
                                continue
                            
                            # Normaliza a expressão
                            expression = str(expression).replace('"', '').strip()
                            
                            # Agrupa por LLM e número da imagem
                            if llm not in results:
                                results[llm] = {}
                            results[llm][image_num] = {
                                'expression': expression,
                                'category': category,
                                'path': filename  # Adiciona o nome do arquivo como 'path'
                            }
                        except json.JSONDecodeError:
                            print(f"Erro ao decodificar o arquivo JSON: {filename}")
                except Exception as e:
                    print(f"Erro ao processar o arquivo {filename}: {e}")
    
    return results, processed_files

def normalize_expression(expr):
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

def identify_operation(expr):
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

def extract_digits(text):
    """Extrai apenas dígitos de um texto"""
    return ''.join(filter(str.isdigit, text))

def evaluate_expression(pred_expr, true_expr):
    """
    Verifica se a expressão predita está correta comparando com a expressão verdadeira
    """
    pred_normalized = normalize_expression(pred_expr)
    true_normalized = normalize_expression(true_expr)
    
    # Verifica igualdade exata após normalização
    exact_match = pred_normalized == true_normalized
    
    # Identifica a operação
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
    
    return {
        'exact_match': exact_match,
        'numbers_match': numbers_match,
        'result_match': result_match,
        'numbers_partial_match': numbers_partial_match,
        'operation': operation
    }

def calculate_metrics(results, ground_truth, llm):
    """
    Calcula as métricas de avaliação para um modelo LLM específico
    """
    llm_results = results.get(llm, {})
    
    # Resultados por imagem
    evaluations = {}
    
    # Contadores agregados
    total = 0
    correct = 0
    operation_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    
    # Listas para calcular precisão, recall e F1
    y_true = []
    y_pred = []
    
    for image_num, true_data in ground_truth.items():
        if image_num in llm_results:
            total += 1
            pred_data = llm_results[image_num]
            pred_expr = pred_data['expression']
            category = pred_data['category']
            path = pred_data.get('path', f"{category}_{llm}_{image_num}_math.json")
            
            # Avalia a expressão
            try:
                evaluation = evaluate_expression(pred_expr, true_data)
                operation = evaluation['operation']
                
                # Atualiza estatísticas da operação
                operation_stats[operation]['total'] += 1
                if evaluation['exact_match']:
                    operation_stats[operation]['correct'] += 1
                    correct += 1
                
                # Para cálculo de precisão/recall/F1
                y_true.append(1)  # O ground truth é sempre correto
                y_pred.append(1 if evaluation['exact_match'] else 0)
                
                # Armazena avaliação detalhada
                evaluations[image_num] = {
                    'path': path,
                    'predicted': pred_expr,
                    'ground_truth': true_data,
                    'category': category,
                    'operation': operation,
                    'exact_match': evaluation['exact_match'],
                    'numbers_match': evaluation['numbers_match'],
                    'result_match': evaluation['result_match'],
                    'numbers_partial_match': evaluation['numbers_partial_match']
                }
            except Exception as e:
                print(f"Erro ao avaliar expressão {pred_expr} vs {true_data}: {e}")
    
    # Calcula métricas
    accuracy = correct / total if total > 0 else 0
    
    # Precisão, recall e F1 (se não houver correspondências, definimos como 0)
    try:
        precision = precision_score(y_true, y_pred) if len(y_pred) > 0 else 0
        recall = recall_score(y_true, y_pred) if len(y_pred) > 0 else 0
        f1 = f1_score(y_true, y_pred) if len(y_pred) > 0 else 0
    except Exception as e:
        print(f"Erro ao calcular métricas para {llm}: {e}")
        precision = recall = f1 = 0
    
    # Estatísticas por operação
    for op, stats in operation_stats.items():
        stats['accuracy'] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'total': total,
        'correct': correct,
        'operation_stats': dict(operation_stats),
        'evaluations': evaluations
    }

def find_missing_images(all_processed_files, ground_truth, results_folders):
    """
    Identifica imagens que estão no ground truth mas não foram processadas
    """
    # Cria um conjunto de todos os números de imagem no ground truth
    ground_truth_numbers = set(ground_truth.keys())
    
    # Cria conjuntos para cada LLM com os números de imagem processados
    processed_numbers = {}
    for llm, llm_results in results.items():
        processed_numbers[llm] = set(llm_results.keys())
    
    # Encontra imagens no ground truth que não foram processadas por cada LLM
    missing_by_llm = {}
    for llm, numbers in processed_numbers.items():
        missing = ground_truth_numbers - numbers
        if missing:
            missing_by_llm[llm] = sorted(list(missing))
    
    # Encontra imagens que não foram processadas por nenhuma LLM
    not_processed_at_all = ground_truth_numbers.copy()
    for numbers in processed_numbers.values():
        not_processed_at_all &= ground_truth_numbers - numbers
    
    # Verifica se todos os arquivos JSON nos diretórios foram processados
    unprocessed_files = {}
    for llm, folder in results_folders.items():
        if os.path.exists(folder):
            all_files = set([f for f in os.listdir(folder) if f.endswith('.json')])
            processed = set(all_processed_files)
            if all_files - processed:
                unprocessed_files[llm] = sorted(list(all_files - processed))
    
    return {
        'missing_by_llm': missing_by_llm,
        'not_processed_at_all': sorted(list(not_processed_at_all)) if not_processed_at_all else [],
        'unprocessed_files': unprocessed_files
    }

def print_results(metrics):
    """
    Imprime os resultados da avaliação de forma organizada
    """
    for llm, llm_metrics in metrics.items():
        print(f"\n{'=' * 50}")
        print(f"RESULTADOS PARA {llm.upper()}")
        print(f"{'=' * 50}")
        
        # Métricas gerais
        print(f"\nMÉTRICAS GERAIS:")
        print(f"  Total de expressões avaliadas: {llm_metrics['total']}")
        print(f"  Total de acertos: {llm_metrics['correct']} ({llm_metrics['accuracy']*100:.2f}%)")
        print(f"  Precisão: {llm_metrics['precision']*100:.2f}%")
        print(f"  Recall: {llm_metrics['recall']*100:.2f}%")
        print(f"  F1 Score: {llm_metrics['f1']*100:.2f}%")
        
        # Métricas por operação
        print(f"\nDESEMPENHO POR OPERAÇÃO:")
        op_stats = llm_metrics['operation_stats']
        for op, stats in sorted(op_stats.items(), key=lambda x: x[1]['accuracy'], reverse=True):
            print(f"  {op.capitalize()}: {stats['correct']}/{stats['total']} acertos ({stats['accuracy']*100:.2f}%)")
        
        # Encontra as expressões erradas
        evaluations = llm_metrics['evaluations']
        errors = [ev for img_num, ev in evaluations.items() if not ev['exact_match']]
        errors_by_expr = defaultdict(int)
        
        for error in errors:
            ground_truth = error['ground_truth']
            errors_by_expr[ground_truth] += 1
        
        if errors:
            print(f"\nEXPRESSÕES MAIS ERRADAS:")
            for expr, count in sorted(errors_by_expr.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"  '{expr}': {count} erros")

def generate_csv_report(metrics, output_file):
    """
    Gera um relatório CSV com os resultados detalhados conforme especificações
    """
    # Cria DataFrame para armazenar os resultados
    rows = []
    
    for llm, llm_metrics in metrics.items():
        evaluations = llm_metrics['evaluations']
        
        for img_num, eval_data in evaluations.items():
            row = {
                'llm': llm,
                'image_number': img_num,
                'path': eval_data['path'],
                'ground_truth': eval_data['ground_truth'],
                'modelo_pred': eval_data['predicted'],
                'category': eval_data['category'],
                'operation': eval_data['operation'],
                'exact_match': eval_data['exact_match'],
                'numbers_match': eval_data['numbers_match'],
                'result_match': eval_data['result_match'],
                'numbers_partial_match': eval_data['numbers_partial_match']
            }
            rows.append(row)
    
    # Cria e salva o DataFrame
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    print(f"\nRelatório detalhado salvo em {output_file}")
    
    # Cria e salva o resumo por LLM
    summary_rows = []
    for llm, llm_metrics in metrics.items():
        summary_row = {
            'llm': llm,
            'total_expressions': llm_metrics['total'],
            'correct': llm_metrics['correct'],
            'accuracy': llm_metrics['accuracy'],
            'precision': llm_metrics['precision'],
            'recall': llm_metrics['recall'],
            'f1': llm_metrics['f1']
        }
        
        # Adiciona estatísticas por operação
        for op, stats in llm_metrics['operation_stats'].items():
            summary_row[f'{op}_total'] = stats['total']
            summary_row[f'{op}_correct'] = stats['correct']
            summary_row[f'{op}_accuracy'] = stats['accuracy']
        
        summary_rows.append(summary_row)
    
    summary_df = pd.DataFrame(summary_rows)
    summary_file = output_file.replace('.csv', '_summary.csv')
    summary_df.to_csv(summary_file, index=False)
    print(f"Resumo por LLM salvo em {summary_file}")

def main():
    # Caminhos atualizados relativos ao diretório atual onde o script está sendo executado
    labels_file = 'matching_labels.txt'  # Arquivo está no mesmo diretório do script
    results_folders = {
        'openai': 'results_openai',      # Pastas estão no mesmo diretório do script
        'google': 'results_google',
        'anthropic': 'results_anthropic'
    }
    output_file = 'math_results_comparison.csv'
    
    # Verifica se os caminhos existem
    if not os.path.exists(labels_file):
        print(f"ERRO: Arquivo de labels não encontrado: {labels_file}")
        print(f"Diretório atual: {os.getcwd()}")
        print(f"Conteúdo do diretório: {os.listdir('.')}")
        
        # Tenta encontrar o arquivo em outros locais
        for root, dirs, files in os.walk(os.getcwd(), topdown=True):
            if 'matching_labels.txt' in files:
                labels_file = os.path.join(root, 'matching_labels.txt')
                print(f"Encontrou arquivo de labels em: {labels_file}")
                break
        
        if not os.path.exists(labels_file):
            return
        
    for llm, folder in results_folders.items():
        if not os.path.exists(folder):
            print(f"ERRO: Pasta de resultados não encontrada para {llm}: {folder}")
    
    # Carrega o ground truth
    print("Carregando ground truth...")
    ground_truth = load_ground_truth(labels_file)
    print(f"Carregado {len(ground_truth)} expressões do ground truth")
    
    # Carrega resultados das LLMs
    global results
    results = {}
    all_processed_files = []
    for llm, folder in results_folders.items():
        print(f"Carregando resultados de {llm}...")
        if not os.path.exists(folder):
            print(f"AVISO: Pasta {folder} não existe. Pulando...")
            results[llm] = {}
            continue
            
        llm_results, processed_files = load_llm_results(folder)
        results[llm] = llm_results.get(llm, {})
        all_processed_files.extend(processed_files)
        print(f"Carregado {len(results[llm])} resultados para {llm}")
    
    # Identifica imagens faltantes
    missing_info = find_missing_images(all_processed_files, ground_truth, results_folders)
    
    # Imprime informações sobre arquivos não processados
    print("\n" + "="*50)
    print("SUMÁRIO DE ARQUIVOS NÃO PROCESSADOS")
    print("="*50)
    
    if missing_info['not_processed_at_all']:
        print(f"\nImagens no ground truth que não foram processadas por nenhuma LLM: {missing_info['not_processed_at_all']}")
    
    for llm, missing in missing_info['missing_by_llm'].items():
        print(f"\nImagens no ground truth não processadas por {llm}: {', '.join(map(str, missing))}")
    
    for llm, unprocessed in missing_info['unprocessed_files'].items():
        if unprocessed:
            print(f"\nArquivos JSON não processados para {llm}:")
            for file in unprocessed[:10]:  # Limite para não sobrecarregar a saída
                print(f"  - {file}")
            if len(unprocessed) > 10:
                print(f"  ... e mais {len(unprocessed) - 10} arquivos")
    
    # Calcula métricas
    print("\nCalculando métricas...")
    metrics = {}
    for llm in results_folders.keys():
        print(f"Avaliando {llm}...")
        metrics[llm] = calculate_metrics(results, ground_truth, llm)
    
    # Imprime resultados resumidos
    print("\n" + "="*50)
    print("SUMÁRIO DE RESULTADOS")
    print("="*50)
    
    for llm, llm_metrics in metrics.items():
        print(f"\n{llm.upper()}:")
        print(f"  Total avaliado: {llm_metrics['total']}")
        print(f"  Acertos: {llm_metrics['correct']} de {llm_metrics['total']} ({llm_metrics['accuracy']*100:.2f}%)")
        print(f"  Precision: {llm_metrics['precision']*100:.2f}%")
        print(f"  Recall: {llm_metrics['recall']*100:.2f}%")
        print(f"  F1 Score: {llm_metrics['f1']*100:.2f}%")
    
    # Gera relatório CSV com o formato solicitado
    generate_csv_report(metrics, output_file)

if __name__ == "__main__":
    main() 
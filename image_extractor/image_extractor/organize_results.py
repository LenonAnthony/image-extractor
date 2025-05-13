#!/usr/bin/env python3
import os
import shutil
import re

def create_result_folders():
    """
    Cria as pastas de resultados se não existirem
    """
    folders = ['results_openai', 'results_anthropic', 'results_google']
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)

def organize_json_results():
    """
    Organiza os arquivos JSON das pastas de resultados dentro de math_images
    """
    # Pastas de origem
    categories = ['horizontais', 'verticais', 'ruins']
    base_path = os.path.join('image_extractor', 'image_extractor', 'math_images')
    
    print(f"Buscando arquivos JSON em: {base_path}")
    if not os.path.exists(base_path):
        print(f"ERRO: Pasta base {base_path} não encontrada!")
        return
    
    # Cria pastas de destino
    create_result_folders()
    
    # Contadores de arquivos
    openai_count = 0
    anthropic_count = 0
    google_count = 0
    unknown_count = 0
    
    # Para cada categoria
    for category in categories:
        category_path = os.path.join(base_path, category)
        if not os.path.exists(category_path):
            print(f"Pasta não encontrada: {category_path}")
            continue
            
        # Verifica se existe a pasta 'results' dentro da categoria
        results_path = os.path.join(category_path, 'results')
        if not os.path.exists(results_path):
            print(f"Pasta de resultados não encontrada em: {category_path}")
            continue
            
        print(f"Processando resultados em: {results_path}")
        
        # Para cada arquivo na pasta de resultados
        for filename in os.listdir(results_path):
            if filename.endswith('.json'):
                src_file = os.path.join(results_path, filename)
                
                # Determina a pasta de destino com base no nome do arquivo
                if 'openai' in filename.lower():
                    dest_folder = 'results_openai'
                    openai_count += 1
                elif 'anthropic' in filename.lower() or 'claude' in filename.lower():
                    dest_folder = 'results_anthropic'
                    anthropic_count += 1
                elif 'google' in filename.lower() or 'gemini' in filename.lower():
                    dest_folder = 'results_google'
                    google_count += 1
                else:
                    print(f"Não foi possível determinar a LLM para o arquivo: {filename}")
                    unknown_count += 1
                    continue
                
                # Copia o arquivo para a pasta de destino
                dst_file = os.path.join(dest_folder, f"{category}_{filename}")
                shutil.copy2(src_file, dst_file)
                print(f"Copiado {filename} de {category}/results para {dest_folder}")
    
    # Imprime estatísticas
    print("\nEstatísticas de arquivos:")
    print(f"Arquivos OpenAI: {openai_count}")
    print(f"Arquivos Anthropic: {anthropic_count}")
    print(f"Arquivos Google: {google_count}")
    print(f"Arquivos desconhecidos: {unknown_count}")
    print(f"Total de arquivos copiados: {openai_count + anthropic_count + google_count}")

def main():
    organize_json_results()
    print("Organização dos resultados JSON completa!")

if __name__ == "__main__":
    main()
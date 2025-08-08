# extraction_main.py

from pathlib import Path
from enum import StrEnum
import click
import time
import json
import re
import threading
# Adicionamos HuggingFaceConversion à lista de imports
from image_extractor.service.text_extraction import (
    OpenAiConversion, 
    VertexAiConversation, 
    GoogleVisionConversion, 
    AnthropicConversion, 
    OllamaConversion, 
    HuggingFaceConversion
)
from image_extractor.model.text_extract import TextExtract
import os

class Model(StrEnum):
    OPENAI = "openai"
    VERTEXAI = "vertexai"
    GOOGLE_VISION = "google_vision"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface" # Nova opção de modelo

class Extension(StrEnum):
    JPG = "jpg"
    JPEG = "jpeg"
    PNG = "png"
    JFIF = "jfif"

def detect_loop(text: str, max_repetitions: int = 5) -> bool:
    """
    Detecta se o texto está em loop baseado em repetições de padrões.
    Retorna True se detectar um loop com pelo menos 5 frases repetidas.
    """
    if not text or len(text) < 50:
        return False
    
    # Divide o texto em frases para análise
    sentences = re.split(r'[.!?]+', text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
    
    if len(sentences) < 5:  # Precisa de pelo menos 5 frases
        return False
    
    # Procura por frases repetidas consecutivamente
    # Conta quantas frases consecutivas são iguais
    consecutive_repetitions = 0
    max_consecutive = 0
    
    for i in range(len(sentences) - 1):
        if sentences[i] == sentences[i + 1]:
            consecutive_repetitions += 1
            max_consecutive = max(max_consecutive, consecutive_repetitions)
        else:
            consecutive_repetitions = 0
    
    # Se há pelo menos 5 frases consecutivas iguais, é um loop
    if max_consecutive >= 4:  # 4 repetições consecutivas = 5 frases iguais
        return True
    
    # Procura por padrões repetitivos (palavras ou frases curtas)
    words = text.split()
    if len(words) > 20:
        # Verifica se há muitas repetições da mesma palavra
        word_counts = {}
        for word in words:
            if len(word) > 3:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Se uma palavra aparece mais de 30% das vezes, pode ser um loop
        total_words = len(words)
        for word, count in word_counts.items():
            if count > total_words * 0.3:
                return True
    
    # Verifica se há repetição de frases completas
    if len(text) > 200:
        # Divide em chunks menores para verificar repetição
        chunk_size = len(text) // 5
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        if len(chunks) >= 2:
            for i in range(len(chunks) - 1):
                if chunks[i] == chunks[i + 1]:
                    return True
    
    return False

def detect_loop_in_stream(text_stream: str) -> bool:
    """
    Detecta loop em texto que está sendo gerado em stream.
    Versão mais sensível para detecção em tempo real.
    Exige pelo menos 5 frases repetidas para detectar loop.
    """
    if not text_stream or len(text_stream) < 50:
        return False
    
    # Verifica repetição de palavras consecutivas
    words = text_stream.split()
    if len(words) > 10:
        # Se a mesma palavra aparece 5 vezes seguidas
        for i in range(len(words) - 4):
            if words[i] == words[i + 1] == words[i + 2] == words[i + 3] == words[i + 4]:
                return True
    
    # Verifica repetição de frases curtas
    sentences = re.split(r'[.!?]+', text_stream.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
    
    if len(sentences) >= 5:
        # Conta frases consecutivas iguais
        consecutive_repetitions = 0
        max_consecutive = 0
        
        for i in range(len(sentences) - 1):
            if sentences[i] == sentences[i + 1]:
                consecutive_repetitions += 1
                max_consecutive = max(max_consecutive, consecutive_repetitions)
            else:
                consecutive_repetitions = 0
        
        # Se há pelo menos 5 frases consecutivas iguais, é um loop
        if max_consecutive >= 4:  # 4 repetições consecutivas = 5 frases iguais
            return True
    
    return False

@click.group()
def cli():
    pass

@cli.command()
@click.option(
    "--folder", prompt="The folder with the images", help="The folder with the images"
)
@click.option(
    "--model",
    default=Model.OPENAI.value,
    prompt="The model you want to use",
    type=click.Choice([m.value for m in Model]),
    help="The model type to be used",
)
@click.option(
    "--extension",
    default="jpg",
    prompt="The extension to process",
    type=click.Choice([e.value for e in Extension]),
    help="The extension of the images",
)
@click.option(
    "--batch_size", default=1, help="The size of the batch"
)
@click.option(
    "--max_time_per_file", 
    default=120,  # Reduzido de 300 para 120 segundos (2 minutos)
    help="Maximum time in seconds to process each file (default: 120s = 2min)"
)

def convert_folder(folder: str, model: str, extension: str, batch_size: int, max_time_per_file: int):
    start = time.time()
    root = Path(folder)
    assert root.exists(), f"Path {root} does not exist."
    
    conversion = None
    if model == Model.OPENAI.value:
        conversion = OpenAiConversion()
    elif model == Model.VERTEXAI.value:
        conversion = VertexAiConversation()
    elif model == Model.GOOGLE_VISION.value:
        conversion = GoogleVisionConversion()
    elif model == Model.ANTHROPIC.value:
        conversion = AnthropicConversion()
    elif model == Model.OLLAMA.value:
        conversion = OllamaConversion()
    elif model == Model.HUGGINGFACE.value: # Adicionamos a lógica para o novo modelo
        conversion = HuggingFaceConversion()
    else:
        raise ValueError(f"Unsupported model type: {model}")

    files = root.rglob(f"**/*.{extension}")

    if model == Model.OPENAI.value:
        model_type = os.getenv("OPENAI_MODEL").replace("gpt-", "")
    elif model == Model.VERTEXAI.value:  
        model_type = os.getenv("GEMINI_MODEL").replace("gemini-", "")
    elif model == Model.ANTHROPIC.value:
        model_type = os.getenv("ANTHROPIC_MODEL").replace("claude-", "")
    elif model == Model.OLLAMA.value:
        model_type = os.getenv("OLLAMA_MODEL", "minicpm-v:8b").replace(":", "_")
    elif model == Model.HUGGINGFACE.value: # Nomenclatura para o arquivo de saída
        default_hf_model = "CEIA-UFG/Gemma-3-Gaia-PT-BR-4b-it"
        model_type = os.getenv("HUGGINGFACE_MODEL", default_hf_model).replace("/", "_")
    else:
        model_type = "unknown"

    def check_existing_file(file_path: Path, model: str, model_type: str) -> bool:
        target_file = file_path.parent / f"{model}_{model_type}_{file_path.stem}.json"
        return target_file.exists()

    for f in files:
        if check_existing_file(f, model, model_type):
            click.echo(f"File {f} was already processed.")
            continue
        
        click.echo(f"Processing file {f}")
        start_file = time.time()
        
        try:
            # Adiciona timeout mais agressivo
            extract = None
            loop_detected = False
            timeout_detected = False
            result_ready = False
            
            def process_with_timeout():
                nonlocal extract, result_ready
                try:
                    extract = conversion.convert_to_text(f)
                    result_ready = True
                except Exception as e:
                    click.echo(f"❌ Erro durante processamento: {str(e)}")
                    extract = None
            
            # Executa o processamento com timeout
            thread = threading.Thread(target=process_with_timeout)
            thread.start()
            
            # Polling para verificar se há resultado disponível
            start_polling = time.time()
            while thread.is_alive() and (time.time() - start_polling) < max_time_per_file:
                time.sleep(0.5)  # Verifica a cada 0.5 segundos
                if result_ready and extract is not None:
                    break
            
            elapsed_file = time.time() - start_file
            
            if thread.is_alive():
                click.echo(f"⏰ Timeout atingido para {f} ({elapsed_file:.1f}s). Tentando capturar resultado parcial...")
                timeout_detected = True
                
                # Se temos algum resultado, usa ele mesmo que incompleto
                if extract is not None:
                    click.echo(f"✅ Capturado resultado parcial para {f}")
                else:
                    click.echo(f"❌ Nenhum resultado parcial capturado para {f}")
                
                # Força a interrupção do thread original
                thread.join(timeout=5)
            else:
                # Processamento completou normalmente
                click.echo(f"✅ Processamento completado para {f}")
            
            # Usa o resultado disponível
            final_result = extract
            
            # Verifica se o resultado está em loop (apenas se não houve timeout)
            if not timeout_detected and final_result and final_result.main_text:
                if detect_loop(final_result.main_text):
                    click.echo(f"⚠️  Loop detectado no arquivo {f}. Salvando resultado parcial e continuando...")
                    loop_detected = True
            
            def process_extract(extract: TextExtract, f: Path, elapsed: float, loop_detected: bool, timeout_detected: bool):
                if extract is not None:
                    target_file = f.parent / f"{model}_{model_type}_{f.stem}.json"
                    extract_data = extract.model_dump()
                    extract_data["elapsed"] = elapsed
                    extract_data["loop_detected"] = loop_detected
                    extract_data["timeout_detected"] = timeout_detected
                    json_data = json.dumps(extract_data, indent=2, ensure_ascii=False)
                    target_file.write_text(json_data, encoding="utf-8")
                    click.echo(f"Wrote {target_file}.")

            process_extract(final_result, f, elapsed_file, loop_detected, timeout_detected)
            
        except Exception as e:
            click.echo(f"❌ Erro ao processar {f}: {str(e)}")
            continue

    end = time.time()
    click.echo(f"Elapsed time: {end - start} seconds.")

if __name__ == "__main__":
    cli()
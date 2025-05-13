from typing import List
import base64
from pathlib import Path
import time
import logging
import random
from functools import wraps
import mimetypes
import httpx
import sys

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from config import cfg
from langchain_openai import ChatOpenAI
from langchain_google_vertexai import ChatVertexAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from model.math_formula_extract import MathFormulaExtract
from google.cloud import vision

# Configuração de logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Definindo a função retry_with_exponential_backoff localmente para evitar problemas de importação
def retry_with_exponential_backoff(
    max_retries=5, 
    base_delay=1, 
    max_delay=60, 
    backoff_factor=2,
    exceptions=(Exception,)
):
    """
    Retry decorator with exponential backoff for handling transient network errors.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            delay = base_delay
            
            while True:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    retries += 1
                    if retries > max_retries:
                        logger.error(f"Maximum retries ({max_retries}) exceeded. Last error: {str(e)}")
                        raise  # Não usar fallback, apenas lança a exceção original
                    
                    # Calculate jitter (random value between 0 and 0.1*delay)
                    jitter = random.uniform(0, 0.1 * delay)
                    sleep_time = min(delay + jitter, max_delay)
                    
                    logger.warning(f"Attempt {retries}/{max_retries} failed with error: {str(e)}. "
                          f"Retrying in {sleep_time:.2f} seconds...")
                    
                    time.sleep(sleep_time)
                    delay = min(delay * backoff_factor, max_delay)
        
        return wrapper
    
    return decorator

PROMPT_INSTRUCTION = (
    """A imagem enviada contém uma operação matemática escrita manualmente, podendo estar disposta em formato vertical (com os números alinhados em colunas, um abaixo do outro), em formato horizontal (linha única), ou com números sobrepostos (indicando "vai 1"). O conteúdo pode estar mal escrito, com dígitos imperfeitos, desalinhados ou pouco legíveis. Mesmo assim, sua tarefa é transcrever fielmente a expressão presente na imagem, convertendo-a para uma única linha horizontal, com todos os dígitos e símbolos separados por espaço. Nunca realize cálulos, nunca corrija erros e nunca interprete o significado dos números — apenas transcreva com exatidão visual o que está representado.

Se os números estiverem dispostos verticalmente, interprete como uma sequência de operações empilhadas e converta para formato horizontal, mantendo a ordem e separando tudo com espaços. Se a operação estiver escrita horizontalmente, apenas transcreva como está, utilizando os símbolos +, -, * para multiplicação(que está como x ou X nas imagens) e / para divisão, sempre com espaço entre cada elemento. Você também nao deve utilizar nenhum outro elemento que não seja numeros ou + ,- ,*,/, . ou , Se houver números escritos acima de outros (indicando "vai 1"), utilize a notação \\overset {{ número_acima }} {{ número_abaixo }} para representar essa sobreposição.

Exemplos de saída correta são:

Copiar
Editar
2 2 5 * 4  
2 2 5 * 2 2 5 * 2 2 5 * 2 2 5 = 8 9 0  
\\overset {{ 1 }} {{ 1 }} \\overset {{ 1 }} {{ 5 }} 6 + 8 9 = 2 4 5
Lembre-se: nunca resolva a equação, nunca corrija o que está manuscrito, e sempre respeite a ordem visual da imagem, separando todos os dígitos e símbolos com espaço."""
)

def convert_base64(image_path: Path) -> (str, str):
    """
    Convert an image file to base64 encoding with accurate MIME type detection.
    
    This function examines the file's magic numbers to determine the actual format,
    which is more reliable than using the file extension.
    """
    bytes_data = image_path.read_bytes()
    
    # Check magic numbers in the file header to get the actual format
    # More reliable than using file extensions or mimetypes.guess_type
    if bytes_data.startswith(b'\xFF\xD8\xFF'):  # JPEG signature
        mime_type = 'image/jpeg'
    elif bytes_data.startswith(b'\x89\x50\x4E\x47\x0D\x0A\x1A\x0A'):  # PNG signature
        mime_type = 'image/png'
    elif bytes_data.startswith(b'\x47\x49\x46\x38'):  # GIF signature
        mime_type = 'image/gif'
    elif bytes_data.startswith(b'\x52\x49\x46\x46') and b'WEBP' in bytes_data[:16]:  # WEBP signature
        mime_type = 'image/webp'
    else:
        # Fallback: try mimetypes
        mime_type, _ = mimetypes.guess_type(str(image_path))
        if mime_type is None:
            # Final fallback: use JPEG as it's most common
            mime_type = 'image/jpeg'
    
    # Convert to base64
    encoded = base64.b64encode(bytes_data).decode("utf-8")
    
    return encoded, mime_type

def create_math_formula_extract_chain(chat_model: BaseChatModel):
    """
    Cria um chain específico para cada tipo de modelo.
    Cada modelo (OpenAI, Anthropic, Google) tem sua própria maneira de formatar mensagens e imagens.
    """
    if isinstance(chat_model, ChatOpenAI):
        # Formato específico do OpenAI
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", PROMPT_INSTRUCTION),
                (
                    "user",
                    [
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:{mime_type};base64,{image_data}"},
                        }
                    ],
                ),
            ]
        )
    elif isinstance(chat_model, ChatVertexAI):
        # Formato específico do Vertex AI (Google Cloud)
        prompt_template = ChatPromptTemplate.from_messages([
            ("user", [
                {"type": "text", "text": PROMPT_INSTRUCTION},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:{mime_type};base64,{image_data}"},
                }
            ])
        ])
    elif isinstance(chat_model, ChatAnthropic):
        # Formato específico do Anthropic
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", PROMPT_INSTRUCTION),
            (
                "user",
                [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:{mime_type};base64,{image_data}"},
                    }
                ],
            ),
        ])
    elif isinstance(chat_model, ChatGoogleGenerativeAI):
        # Formato específico do Google Generative AI (Gemini)
        # Pré-formatamos a URL da imagem para evitar múltiplas variáveis de formatação
        # que não são suportadas pelo modelo Google
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "Você é um especialista em extrair fórmulas matemáticas de imagens."),
            ("user", [
                {"type": "text", "text": PROMPT_INSTRUCTION + "\n\nAnalise a imagem anexada."},
                {"type": "image_url", "image_url": "{image_url}"}
            ])
        ])
    else:
        raise ValueError(f"Model type {type(chat_model)} not supported")

    # Cria o chain com o formato adequado para estruturar a saída
    try:
        chain = prompt_template | chat_model.with_structured_output(MathFormulaExtract)
        return chain
    except Exception as e:
        # Se falhar com saída estruturada, tenta com saída de texto simples
        print(f"Erro ao criar chain estruturado: {str(e)}. Tentando formato simples.")
        return prompt_template | chat_model

def execute_math_formula_prompt_internal(chat_model: BaseChatModel, image_path: Path) -> MathFormulaExtract:
    """Internal implementation without retry logic"""
    start = time.time()
    try:
        converted_img, mime_type = convert_base64(image_path)
        chain = create_math_formula_extract_chain(chat_model)
        
        # Pre-format the image URL for Google Generative AI model
        if isinstance(chat_model, ChatGoogleGenerativeAI):
            # Create a pre-formatted URL for Google model that only accepts one format variable
            image_url = f"data:{mime_type};base64,{converted_img}"
            result = chain.invoke({"image_url": image_url})
        else:
            # For other models, use the regular format with separate variables
            result = chain.invoke({"image_data": converted_img, "mime_type": mime_type})
            
        elapsed = time.time() - start
        
        # Verifica se o resultado é None antes de tentar acessar atributos
        if result is None:
            print(f"Model returned None for {image_path}")
            return MathFormulaExtract(
                formula=f"ERROR: Model returned None result",
                elapsed_time=elapsed
            )
        
        # Se o resultado é uma string (caso não tenha sido estruturado)
        if isinstance(result, str):
            print(f"Result is a string rather than structured output. Converting.")
            return MathFormulaExtract(
                formula=result.strip(),
                elapsed_time=elapsed
            )
        
        # Garante que o tempo está no resultado
        if hasattr(result, 'model_copy'):
            result = result.model_copy(update={"elapsed_time": elapsed})
        else:
            try:
                result.elapsed_time = elapsed
            except (AttributeError, TypeError):
                # Se não conseguir adicionar o atributo, criar um novo objeto com os dados do resultado
                formula = getattr(result, 'formula', str(result)) if result is not None else "ERROR: Empty result"
                return MathFormulaExtract(
                    formula=formula,
                    elapsed_time=elapsed
                )
        
        return result
    except Exception as e:
        print(f"Error in execute_math_formula_prompt_internal: {str(e)}")
        # Retorna um objeto de erro para não quebrar a execução
        return MathFormulaExtract(
            formula=f"ERROR: {str(e)}",
            elapsed_time=time.time() - start
        )

@retry_with_exponential_backoff(
    max_retries=5,  # Aumentando o número de tentativas
    base_delay=3,   # Aumentando o delay inicial
    max_delay=60,   # Aumentando o delay máximo
    backoff_factor=2,
    exceptions=(httpx.ReadError, httpx.ConnectError, httpx.TimeoutException, Exception)
)
def execute_math_formula_prompt(chat_model: BaseChatModel, image_path: Path) -> MathFormulaExtract:
    """Execute prompt with retry mechanism (sem fallback para outros modelos)"""
    logger.info(f"Processing image {image_path} with model {type(chat_model).__name__}")
    return execute_math_formula_prompt_internal(chat_model, image_path)

class MathFormulaAiConversion:
    def __init__(self, model):
        self.model = model
        self.model_name = type(model).__name__
        logger.info(f"Initialized {self.__class__.__name__} with model {self.model_name}")

    def convert_to_formula(self, image_path: Path) -> MathFormulaExtract:
        logger.info(f"Converting image {image_path} using {self.model_name}")
        try:
            return execute_math_formula_prompt(self.model, image_path)
        except Exception as e:
            logger.error(f"Error in convert_to_formula using {self.model_name}: {str(e)}")
            # Retorna um erro sem tentar outro modelo
            return MathFormulaExtract(
                formula=f"ERROR: {str(e)}",
                elapsed_time=0.0
            )

class OpenAiMathFormulaConversion(MathFormulaAiConversion):
    def __init__(self):
        super().__init__(cfg.chat_openai)

class VertexAiMathFormulaConversion(MathFormulaAiConversion):
    def __init__(self):
        super().__init__(cfg.gemini_model)

class AnthropicMathFormulaConversion(MathFormulaAiConversion):
    def __init__(self):
        super().__init__(cfg.chat_anthropic)

class GoogleMathFormulaConversion(MathFormulaAiConversion):
    def __init__(self):
        super().__init__(cfg.gemini_model)
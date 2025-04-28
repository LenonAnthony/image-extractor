from typing import List, Protocol, Dict
import base64
from pathlib import Path
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
import sys
import os

# Adicionar o diretório pai ao caminho do Python para permitir importações absolutas
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import cfg
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from model.answer_sheet_extract import AnswerSheetExtract, AnswerSheetExtractWithImage, BoundingBox, DetectionItem

PROMPT_INSTRUCTION = """
Analise a imagem de um gabarito de prova e retorne um JSON contendo apenas o campo "detections".
Cada entrada em "detections" deve ter a chave igual ao número da questão e o valor um objeto com:
- "name": opção marcada (A, B, C, D ou "invalid")
- "bounding_box": coordenadas normalizadas (0.0–1.0) com campos x1, y1, x2, y2
- "confidence": valor em float (0.0–1.0)

Exemplo:
{
  "detections": {
    "1": {
      "name": "C",
      "bounding_box": {
        "x1": 0.110402,
        "y1": 0.014023,
        "x2": 0.468166,
        "y2": 0.103992
      },
      "confidence": 0.958050
    },
    "2": {
      "name": "B",
      "bounding_box": {
        "x1": 0.110897,
        "y1": 0.103425,
        "x2": 0.468326,
        "y2": 0.193170
      },
      "confidence": 0.929779
    }
  }
}
"""

def convert_base64(image_path: Path) -> str:
    bytes = image_path.read_bytes()
    return base64.b64encode(bytes).decode("utf-8")

def create_answer_sheet_chain(chat_model: BaseChatModel):
    """Creates a chain for extracting answers from answer sheets using different LLM providers."""
    # mesmo prompt para todos os modelos
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", PROMPT_INSTRUCTION),
        ("user", [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{image_data}"}}])
    ])

    return prompt_template | chat_model.with_structured_output(AnswerSheetExtract)

def execute_structured_prompt(
    chat_model: BaseChatModel, image_path: Path
) -> AnswerSheetExtract:
    """Execute the prompt to extract answers from a single answer sheet image."""
    converted_img = convert_base64(image_path)
    chain = create_answer_sheet_chain(chat_model)
    return chain.invoke({"image_data": converted_img})

def execute_batch_structured_prompt(
    chat_model: BaseChatModel, image_paths: List[Path], batch_size: int
) -> List[AnswerSheetExtractWithImage]:
    """Process multiple answer sheet images in batches."""
    if batch_size < 0:
        batch_size = 1
    batches = [
        image_paths[i : i + batch_size] for i in range(len(image_paths))[::batch_size]
    ]
    chain = create_answer_sheet_chain(chat_model)
    res: List[AnswerSheetExtractWithImage] = []
    
    for b in batches:
        extracts = chain.batch(
            [{"image_data": convert_base64(img)} for img in b]
        )
        
        for path, extract in zip(b, extracts):
            res.append(
                AnswerSheetExtractWithImage(
                    path=path,
                    answers=extract.answers,
                    total_questions=extract.total_questions,
                    sheet_id=extract.sheet_id
                )
            )
    
    return res

class AnswerSheetExtractor(Protocol):
    """Protocol for answer sheet extractor implementations."""
    def extract_answers(self, image_path: Path) -> AnswerSheetExtract:
        """Extract answers from an image."""
        ...

class OpenAiAnswerSheetExtractor:
    def extract_answers(self, image_path: Path) -> AnswerSheetExtract:
        """Extract answers from an image using OpenAI."""
        return execute_structured_prompt(cfg.chat_openai, image_path)

class VertexAiAnswerSheetExtractor:
    def extract_answers(self, image_path: Path) -> AnswerSheetExtract:
        """Extract answers from an image using Google Gemini."""
        return execute_structured_prompt(cfg.gemini_model, image_path)

class AnthropicAnswerSheetExtractor:
    def extract_answers(self, image_path: Path) -> AnswerSheetExtract:
        """Extract answers from an image using Anthropic."""
        try:
            # Use uma versão personalizada que não depende do structured_output
            converted_img = convert_base64(image_path)
            anthropic_prompt = """Analise a imagem do gabarito e identifique as questões marcadas.
Para cada questão, indique o número, a opção marcada (A, B, C, D ou 'invalid') e sua confiança.
Forneça também as coordenadas (bounding box) de cada questão no formato normalizado (valores entre 0 e 1).

Retorne o resultado neste formato JSON exato:

{
  "answers": [
    {
      "question_number": 1,
      "marked_option": "C",
      "confidence": 0.95,
      "bounding_box": {
        "x1": 0.11,
        "y1": 0.01,
        "x2": 0.46,
        "y2": 0.10
      }
    },
    {
      "question_number": 2,
      "marked_option": "B",
      "confidence": 0.93,
      "bounding_box": {
        "x1": 0.11,
        "y1": 0.10,
        "x2": 0.46,
        "y2": 0.19
      }
    }
  ],
  "total_questions": 2
}"""
            
            # Criar o prompt sem usar o structured_output
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", anthropic_prompt),
                ("user", [{
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{converted_img}"},
                }])
            ])
            
            # Obter a resposta direta como string
            chain = prompt_template | cfg.chat_anthropic
            result_text = chain.invoke({"image_data": converted_img})
            
            # Extrair o JSON da resposta
            json_str = result_text.content
            
            # Encontrar o JSON na resposta (caso tenha texto antes ou depois)
            import re
            json_match = re.search(r'\{[\s\S]*\}', json_str)
            if json_match:
                json_str = json_match.group(0)
            
            # Converter para dicionário
            import json
            result_dict = json.loads(json_str)
            
            # Criar o objeto AnswerSheetExtract usando o novo campo `detections`
            detections: Dict[str, DetectionItem] = {}
            for ans in result_dict.get("answers", []):
                qnum = str(ans.get("question_number", 0))
                # define bounding box
                bbox_data = ans.get("bounding_box", {})
                bbox = BoundingBox(
                    x1=float(bbox_data.get("x1", 0.1)),
                    y1=float(bbox_data.get("y1", 0.1)),
                    x2=float(bbox_data.get("x2", 0.4)),
                    y2=float(bbox_data.get("y2", 0.2))
                )
                detections[qnum] = DetectionItem(
                    name=str(ans.get("marked_option", "invalid")),
                    confidence=float(ans.get("confidence", 0.5)),
                    bounding_box=bbox
                )
            return AnswerSheetExtract(
                detections=detections,
                sheet_id=image_path.stem
            )
        except Exception as e:
            print(f"Error processing with Anthropic: {str(e)}")
            # Failback para retornar estrutura vazia
            return AnswerSheetExtract(detections={}, sheet_id=image_path.stem)
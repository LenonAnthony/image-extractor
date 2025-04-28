from typing import List, Dict, Any
import base64
import json
import re
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from image_extractor.config import cfg
from langchain_openai import ChatOpenAI
from langchain_google_vertexai import ChatVertexAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from image_extractor.model.answer_sheet_extract import AnswerSheetExtract, AnswerSheetExtractWithImage, BoundingBox, QuestionAnswer

PROMPT_INSTRUCTION = """Analyze the image of an answer sheet and identify the questions and marked answers.
For each question, determine:

1. The question number
2. Which option was marked (A, B, C, D, or 'invalid' if multiple options were marked or if no option was marked)
3. The bounding box coordinates of the entire question area (not just the marked option)
4. A confidence score between 0 and 1 for your identification

Please return a structured JSON response with EXACTLY the following format:
{{
  "total_questions": number,
  "questions": [
    {{
      "question_number": number,
      "selected_option": "A" | "B" | "C" | "D" | "invalid",
      "confidence": number,
      "bounding_box": {{
        "x1": number,
        "y1": number,
        "x2": number,
        "y2": number
      }}
    }},
    ...
  ]
}}

Where:
- x1, y1: coordinates of the top-left corner of the question area
- x2, y2: coordinates of the bottom-right corner of the question area
- All coordinates should be normalized between 0 and 1

Do not include any other fields in the response. Follow this exact format."""

def convert_base64(image_path: Path) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def create_answer_sheet_extract_chain(chat_model: BaseChatModel):
    if isinstance(chat_model, ChatOpenAI):
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", PROMPT_INSTRUCTION),
                (
                    "user",
                    [
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/jpeg;base64,{image_data}"},
                        }
                    ],
                ),
            ]
        )
    elif isinstance(chat_model, ChatVertexAI):
        prompt_template = ChatPromptTemplate.from_messages([
            ("user", [
                {"type": "text", "text": PROMPT_INSTRUCTION},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,{image_data}"},
                }
            ])
        ])
    elif isinstance(chat_model, ChatAnthropic):
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", PROMPT_INSTRUCTION),
            (
                "user",
                [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,{image_data}"},
                    }
                ],
            ),
        ])
    elif isinstance(chat_model, ChatGoogleGenerativeAI):
        # O Google Gemini exige que sempre haja um parâmetro de texto
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", PROMPT_INSTRUCTION),
            (
                "user",
                [
                    {"type": "text", "text": "Analyze this answer sheet image:"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,{image_data}"},
                    }
                ],
            ),
        ])
    else:
        raise ValueError(f"Model type {type(chat_model)} not supported")

    # Usar text output em vez de structured_output
    return prompt_template | chat_model


def extract_json_from_text(text: str) -> Dict[str, Any]:
    """Extract JSON from text response that might contain markdown or other text."""
    # Try to find JSON block in markdown
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if json_match:
        json_str = json_match.group(1)
    else:
        # If no markdown code block, try to extract anything that looks like JSON
        json_str = text
        
    # Clean up the string to ensure it's valid JSON
    # Remove any non-JSON content at the beginning and end
    start_idx = json_str.find('{')
    end_idx = json_str.rfind('}') + 1
    
    if start_idx != -1 and end_idx != 0:
        json_str = json_str[start_idx:end_idx]
        
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        raise ValueError(f"Failed to parse JSON from response: {text}")


def parse_answer_sheet_extract(data: Dict[str, Any]) -> AnswerSheetExtract:
    """Parse the JSON data into an AnswerSheetExtract object."""
    questions = []
    
    for q in data.get('questions', []):
        bbox = q.get('bounding_box', {})
        questions.append(
            QuestionAnswer(
                question_number=q.get('question_number', 0),
                selected_option=q.get('selected_option', 'invalid'),
                confidence=q.get('confidence', 0.0),
                bounding_box=BoundingBox(
                    x1=bbox.get('x1', 0.0),
                    y1=bbox.get('y1', 0.0),
                    x2=bbox.get('x2', 0.0),
                    y2=bbox.get('y2', 0.0)
                )
            )
        )
    
    return AnswerSheetExtract(
        total_questions=data.get('total_questions', len(questions)),
        questions=questions
    )


def execute_answer_sheet_extraction(
    chat_model: BaseChatModel, image_path: Path
) -> AnswerSheetExtract:
    converted_img = convert_base64(image_path)
    chain = create_answer_sheet_extract_chain(chat_model)
    
    # Get raw text response instead of structured data
    response = chain.invoke({"image_data": converted_img})
    
    # Extract and parse JSON from response
    json_data = extract_json_from_text(response.content)
    return parse_answer_sheet_extract(json_data)


def execute_batch_answer_sheet_extraction(
    chat_model: BaseChatModel, image_paths: List[Path], batch_size: int
) -> List[AnswerSheetExtractWithImage]:
    if batch_size < 0:
        batch_size = 1
    batches = [
        image_paths[i : i + batch_size] for i in range(len(image_paths))[::batch_size]
    ]
    chain = create_answer_sheet_extract_chain(chat_model)
    res: List[AnswerSheetExtractWithImage] = []
    
    for b in batches:
        responses = chain.batch([{"image_data": convert_base64(img)} for img in b])
        
        for path, response in zip(b, responses):
            try:
                # Extract and parse JSON from response
                json_data = extract_json_from_text(response.content)
                extract = parse_answer_sheet_extract(json_data)
                
                res.append(
                    AnswerSheetExtractWithImage(
                        path=str(path),
                        total_questions=extract.total_questions,
                        questions=extract.questions
                    )
                )
            except Exception as e:
                print(f"Error processing {path}: {e}")
                
    return res


class AnswerSheetExtractor:
    def __init__(self, model):
        self.model = model

    def extract_answer_sheet(self, image_path: Path) -> AnswerSheetExtract:
        return execute_answer_sheet_extraction(self.model, image_path)

    def extract_answer_sheets_batch(
        self, image_paths: List[Path], batch_size: int
    ) -> List[AnswerSheetExtractWithImage]:
        return execute_batch_answer_sheet_extraction(self.model, image_paths, batch_size)


class OpenAiAnswerSheetExtractor(AnswerSheetExtractor):
    def __init__(self):
        super().__init__(cfg.chat_openai)


class VertexAiAnswerSheetExtractor(AnswerSheetExtractor):
    def __init__(self):
        super().__init__(cfg.vertexai_gemini)


class GoogleGenAiAnswerSheetExtractor(AnswerSheetExtractor):
    def __init__(self):
        super().__init__(cfg.gemini_model)


class AnthropicAnswerSheetExtractor(AnswerSheetExtractor):
    def __init__(self):
        super().__init__(cfg.chat_anthropic)

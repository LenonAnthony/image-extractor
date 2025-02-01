from typing import List
import base64
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from image_extractor.config import cfg
from langchain_openai import ChatOpenAI
from langchain_google_vertexai import ChatVertexAI
from image_extractor.model.text_extract import TextExtract, TextExtractWithImage
from google.cloud import vision

PROMPT_INSTRUCTION = """Extraia o texto da imagem. Considere o idioma Português."""

def convert_base64(image_path: Path) -> str:
    bytes = image_path.read_bytes()
    return base64.b64encode(bytes).decode("utf-8")

def create_text_extract_chain(chat_model: BaseChatModel):
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
    else:
        raise ValueError(f"Model type {type(chat_model)} not supported")

    return prompt_template | chat_model.with_structured_output(TextExtract)

def execute_structured_prompt(
    chat_model: BaseChatModel, image_path: Path
) -> TextExtract:
    converted_img = convert_base64(image_path)
    chain = create_text_extract_chain(chat_model)
    return chain.invoke({"image_data": converted_img})

def execute_batch_structured_prompt(
    chat_model: BaseChatModel, image_paths: List[Path], batch_size: int
) -> List[TextExtractWithImage]:
    if batch_size < 0:
        batch_size = 1
    batches = [
        image_paths[i : i + batch_size] for i in range(len(image_paths))[::batch_size]
    ]
    chain = create_text_extract_chain(chat_model)
    res: List[TextExtract] = []
    for b in batches:
        extracts: List[TextExtract] = chain.batch(
            [{"image_data": convert_base64(img)} for img in b]
        )
        for path, extract in zip(b, extracts):
            res.append(
                TextExtractWithImage(
                    path=path,
                    title=extract.title,
                    main_text=extract.main_text,
                    main_text_en=extract.main_text_en,
                    objects_in_image=extract.objects_in_image,
                )
            )
    return res

class AiConversion:
    def __init__(self, model):
        self.model = model

    def convert_to_text(self, image_path: Path) -> TextExtract:
        return execute_structured_prompt(self.model, image_path)

    def convert_to_text_batches(
        self, image_paths: List[Path], batch_size: int
    ) -> List[TextExtractWithImage]:
        return execute_batch_structured_prompt(self.model, image_paths, batch_size)

class OpenAiConversion(AiConversion):
    def __init__(self):
        super().__init__(cfg.chat_openai)

class VertexAiConversation(AiConversion):
    def __init__(self):
        super().__init__(cfg.vertexai_gemini)

class GoogleVisionConversion(AiConversion):
    def __init__(self):
        super().__init__(cfg.google_vision)

    def convert_to_text(self, image_path: Path) -> TextExtract:
        client = self.model
        with open(image_path, "rb") as image_file:
            content = image_file.read()
        image = vision.Image(content=content)
        response = client.text_detection(image=image)
        texts = response.text_annotations

        if texts:
            extracted_text = texts[0].description
        else:
            extracted_text = ""

        if response.error.message:
            raise Exception(
                f"{response.error.message}\nFor more info on error messages, check: "
                "https://cloud.google.com/apis/design/errors"
            )

        return TextExtract(
            main_text=extracted_text,
        )


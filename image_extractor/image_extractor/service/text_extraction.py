from typing import List
import base64
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from image_extractor.config import cfg
from langchain_openai import ChatOpenAI
from langchain_google_vertexai import ChatVertexAI
from image_extractor.model.text_extract import TextExtract, TextExtractWithImage

PROMPT_INSTRUCTION = """Please accurately extract all visible text written in Portuguese (Brazil) from the provided image,
focusing specifically on elementary school-level essays. Ensure the extraction includes complete sentences, proper formatting (e.g., paragraphs, punctuation),
and ignores non-text elements (e.g., images, doodles, watermarks). Prioritize educational content such as student essays, writing exercises, or school assignments. Confirm that the extracted text is in its
original language (Portuguese) and relevant to elementary education in Brazil."""

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


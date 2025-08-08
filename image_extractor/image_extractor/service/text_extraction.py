from typing import List
import base64
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from image_extractor.config import cfg
from langchain_openai import ChatOpenAI
from langchain_google_vertexai import ChatVertexAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from image_extractor.model.text_extract import TextExtract, TextExtractWithImage
from google.cloud import vision
from PIL import Image

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
    elif isinstance(chat_model, ChatOllama):
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

class AnthropicConversion(AiConversion):
    def __init__(self):
        super().__init__(cfg.chat_anthropic)

class OllamaConversion(AiConversion):
    def __init__(self):
        super().__init__(cfg.chat_ollama)

class HuggingFaceConversion(AiConversion):
    def __init__(self):
        super().__init__({
            'model': cfg.hf_model,
            'processor': cfg.hf_processor
        })

    def convert_to_text(self, image_path: Path) -> TextExtract:
        """
        Implementação de baixo nível que espelha o script de teste funcional,
        bypassing a `pipeline()` do transformers.
        """
        model = self.model['model']
        processor = self.model['processor']
        
        if not model or not processor:
            raise ValueError("Modelo ou processador Hugging Face não inicializado. Verifique a config.")

        # 1. Carregar a imagem
        image = Image.open(image_path).convert("RGB")

        # 2. Construir o prompt com o template de chat
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT_INSTRUCTION}]}
        ]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # 3. Processar entradas (texto e imagem)
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)

        # 4. Gerar a saída do modelo
        output = model.generate(**inputs, max_new_tokens=2048) # Aumentado para textos longos de redações

        # 5. Decodificar e limpar a resposta
        response = processor.decode(output[0], skip_special_tokens=True)
        
        # Lógica de limpeza para extrair apenas a resposta do modelo
        model_response_start = response.find("<start_of_turn>model\n")
        if model_response_start != -1:
            final_response = response[model_response_start + len("<start_of_turn>model\n"):].strip()
        else:
            final_response = response.strip()
            
        # Retorna o objeto TextExtract, preenchendo apenas o campo principal
        return TextExtract(main_text=final_response)

    def convert_to_text_batches(
        self, image_paths: List[Path], batch_size: int
    ) -> List[TextExtractWithImage]:
        """
        Implementação de lote que itera sobre as imagens, chamando a lógica
        de conversão individual que agora funciona corretamente.
        """
        res = []
        for path in image_paths:
            try:
                extract = self.convert_to_text(path)
                # Cria o objeto de resultado com os campos disponíveis
                res.append(
                    TextExtractWithImage(
                        path=path,
                        main_text=extract.main_text
                    )
                )
            except Exception as e:
                print(f"Failed to process {path}: {e}")
                continue
        return res
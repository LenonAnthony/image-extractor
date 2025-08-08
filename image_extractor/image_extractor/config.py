# config.py

import os
import torch
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_vertexai import ChatVertexAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from google.cloud import vision
torch.set_float32_matmul_precision('high')
# Importações necessárias para o carregamento manual do modelo Hugging Face
from transformers import AutoProcessor, AutoModelForCausalLM

load_dotenv()

os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"
os.environ["GRPC_POLL_STRATEGY"] = "poll"

class Config:
    def __init__(self):
        # OpenAI Configuration
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_model = os.getenv("OPENAI_MODEL")
        
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.anthropic_model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")

        self.project_id = os.getenv("PROJECT_ID")
        self.location = os.getenv("LOCATION")
        self.gemini_model = os.getenv("GEMINI_MODEL")

        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "minicpm-v:8b")

        self.huggingface_api_token = os.getenv("HUGGINGFACE_API_TOKEN", "")
        self.huggingface_model = os.getenv("HUGGINGFACE_MODEL", "CEIA-UFG/Gemma-3-Gaia-PT-BR-4b-it")

    @property
    def chat_openai(self):
        if self.openai_api_key and self.openai_model:
            return ChatOpenAI(model=self.openai_model, api_key=self.openai_api_key)
        return None

    @property
    def chat_anthropic(self):
        if self.anthropic_api_key:
            return ChatAnthropic(model=self.anthropic_model, api_key=self.anthropic_api_key)
        return None
    
    @property
    def vertexai_gemini(self):
        if self.project_id and self.gemini_model:
            return ChatVertexAI(model_name=self.gemini_model, project=self.project_id, location=self.location)
        return None

    @property
    def google_vision(self):
        return vision.ImageAnnotatorClient()
    
    @property
    def chat_ollama(self):
        return ChatOllama(model=self.ollama_model, base_url=self.ollama_base_url)
    
    @property
    def hf_model(self):
        """Carrega e retorna o modelo Hugging Face."""
        if self.huggingface_model and self.huggingface_api_token:
            print("Carregando modelo Hugging Face...")
            model = AutoModelForCausalLM.from_pretrained(
                self.huggingface_model,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                token=self.huggingface_api_token
            )
            print("Modelo carregado.")
            return model
        return None
        
    @property
    def hf_processor(self):
        """Carrega e retorna o processador Hugging Face."""
        if self.huggingface_model and self.huggingface_api_token:
            print("Carregando processador Hugging Face...")
            processor = AutoProcessor.from_pretrained(
                self.huggingface_model,
                use_fast=True,
                token=self.huggingface_api_token
            )
            print("Processador carregado.")
            return processor
        return None

cfg = Config()
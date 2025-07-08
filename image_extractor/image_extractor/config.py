import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_vertexai import ChatVertexAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from google.cloud import vision
load_dotenv()

os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"
os.environ["GRPC_POLL_STRATEGY"] = "poll"

class Config:
    def __init__(self):
        # OpenAI Configuration
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_model = os.getenv("OPENAI_MODEL")
        
        # Anthropic Configuration
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.anthropic_model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")
        
        # Google/Vertex AI Configuration
        self.project_id = os.getenv("PROJECT_ID")
        self.location = os.getenv("LOCATION")
        self.gemini_model = os.getenv("GEMINI_MODEL")
        
        # Ollama Configuration
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "qwen2.5vl:7b")

    @property
    def chat_openai(self):
        """Instancia ChatOpenAI apenas quando necessário"""
        if self.openai_api_key and self.openai_model:
            return ChatOpenAI(model=self.openai_model, api_key=self.openai_api_key)
        return None

    @property
    def chat_anthropic(self):
        """Instancia ChatAnthropic apenas quando necessário"""
        if self.anthropic_api_key:
            return ChatAnthropic(model=self.anthropic_model, api_key=self.anthropic_api_key)
        return None
    
    @property
    def vertexai_gemini(self):
        """Instancia ChatVertexAI apenas quando necessário"""
        if self.project_id and self.gemini_model:
            return ChatVertexAI(model_name=self.gemini_model, 
                           project=self.project_id, 
                           location=self.location)
        return None

    @property
    def google_vision(self):
        """Instancia Google Vision client"""
        return vision.ImageAnnotatorClient()
    
    @property
    def chat_ollama(self):
        """Instancia ChatOllama apenas quando necessário"""
        return ChatOllama(
            model=self.ollama_model,
            base_url=self.ollama_base_url
        )

cfg = Config()
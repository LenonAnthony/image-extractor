import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from google.cloud import vision
load_dotenv()

class Config:
    def __init__(self):
        # OpenAI settings
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        assert self.openai_api_key, "OpenAI API key is required"
        self.openai_model = os.getenv("OPENAI_MODEL")
        assert self.openai_model, "OpenAI model is required"
        
        # Anthropic settings
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        assert self.anthropic_api_key, "Anthropic API key is required"
        self.anthropic_model = os.getenv("ANTHROPIC_MODEL", "claude-3-7-sonnet-latest")
        assert self.anthropic_model, "Anthropic model is required"
        
        # Google Gemini settings
        self.gemini_model_name = os.getenv("GEMINI_MODEL")
        assert self.gemini_model_name, "Gemini model is required"
        
        # Set Google API Key for environment
        os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
        assert os.environ["GOOGLE_API_KEY"], "Google API Key is required"

    @property
    def chat_openai(self):
        """Instancia `ChatOpenAI` apenas quando necessário"""
        return ChatOpenAI(model=self.openai_model, api_key=self.openai_api_key,temperature=0.0)

    @property
    def chat_anthropic(self):
        """Instancia `ChatAnthropic` apenas quando necessário"""
        if self.anthropic_api_key:
            return ChatAnthropic(model=self.anthropic_model, api_key=self.anthropic_api_key, max_tokens=2000, temperature=0.0)
        return None

    @property
    def gemini_model(self):
        """Instancia `ChatGoogleGenerativeAI` apenas quando necessário"""
        return ChatGoogleGenerativeAI(model=self.gemini_model_name,temperature=0.0)

    @property
    def google_vision(self):
        return vision.ImageAnnotatorClient()


cfg = Config()
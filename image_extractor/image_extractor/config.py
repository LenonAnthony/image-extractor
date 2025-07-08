import os
from langchain_openai import ChatOpenAI
<<<<<<< Updated upstream
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv

=======
from langchain_google_vertexai import ChatVertexAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from google.cloud import vision
>>>>>>> Stashed changes
load_dotenv()


class Config:
    open_ai_key = os.getenv("OPENAI_API_KEY")
    assert open_ai_key is not None, "There is no Open AI key"
    open_ai_model = os.getenv("OPENAI_MODEL")
    assert open_ai_model is not None, "Please specify your OpenAI model"
    chat_open_ai = ChatOpenAI(model=open_ai_model, api_key=open_ai_key)

<<<<<<< Updated upstream
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    assert gemini_api_key is not None, "Cannot find Gemini API key"
    google_model = os.getenv("GOOGLE_MODEL")
    assert google_model is not None, "Please specify your Google Gemini model"
    google_ai = ChatGoogleGenerativeAI(model=google_model, api_key=gemini_api_key)


cfg = Config()
=======
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_model = os.getenv("OPENAI_MODEL")
        
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.anthropic_model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")
        
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "qwen2.5vl:7b")


    @property
    def chat_openai(self):
        """Instancia `ChatOpenAI` apenas quando necessário"""
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required")
        if not self.openai_model:
            raise ValueError("OpenAI model is required")
        return ChatOpenAI(model=self.openai_model, api_key=self.openai_api_key)

    @property
    def chat_anthropic(self):
        """Instancia `ChatAnthropic` apenas quando necessário"""
        if not self.anthropic_api_key:
            raise ValueError("Anthropic API key is required")
        if not self.anthropic_model:
            raise ValueError("Anthropic model is required")
        return ChatAnthropic(model=self.anthropic_model, api_key=self.anthropic_api_key)

    @property
    def chat_ollama(self):
        """Instancia `ChatOllama` apenas quando necessário"""
        if not self.ollama_base_url:
            raise ValueError("Ollama base URL is required")
        if not self.ollama_model:
            raise ValueError("Ollama model is required")
        return ChatOllama(model=self.ollama_model, base_url=self.ollama_base_url)

    #     project_id = os.getenv("PROJECT_ID")
    #     assert project_id, "Project ID is required"
    #     location = os.getenv("LOCATION")
    #     assert location, "Location is required"
    #     gemini_model = os.getenv("GEMINI_MODEL")
    #     assert gemini_model, "Gemini model is required"

    # @property
    # def vertexai_gemini(self):
    #     if self.project_id:
    #         return ChatVertexAI(model_name=self.gemini_model, 
    #                        project=self.project_id, 
    #                        location=self.location)
    #     return None

    @property
    def google_vision(self):
        return vision.ImageAnnotatorClient()

cfg = Config()
>>>>>>> Stashed changes

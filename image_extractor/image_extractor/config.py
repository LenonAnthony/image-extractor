import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_vertexai import ChatVertexAI, GemmaVertexAIModelGarden
from langchain_google_vertexai import GemmaLocalHF, GemmaChatLocalHF
from langchain_anthropic import ChatAnthropic
from google.cloud import vision
from langchain_mistralai import ChatMistralAI    # novo import
load_dotenv()

os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"
os.environ["GRPC_POLL_STRATEGY"] = "poll"

class Config:
    # quando for utilizar os modelos da openAI, tira os comentários abaixo de __init__ e chat_openai e
    #  comentar a parte do gemini (de project_id até o fim da função def vertexai, google vision ...etc)

    # def __init__(self):
    #     self.openai_api_key = os.getenv("OPENAI_API_KEY")
    #     assert self.openai_api_key, "OpenAI API key is required"
    #     self.openai_model = os.getenv("OPENAI_MODEL")
    #     assert self.openai_model, "OpenAI model is required"


    # @property
    # def chat_openai(self):
    #     """Instancia `ChatOpenAI` apenas quando necessário"""
    #     return ChatOpenAI(model=self.openai_model, api_key=self.openai_api_key)      
            
    #     self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    #     assert self.anthropic_api_key, "Anthropic API key is required"
    #     self.anthropic_model = os.getenv("ANTHROPIC_MODEL", "claude-3-7-sonnet-latest")
    #     assert self.anthropic_model, "Anthropic model is required"

    # @property
    # def chat_anthropic(self):
    #     """Instancia `ChatAnthropic` apenas quando necessário"""
    #     if self.anthropic_api_key:
    #         return ChatAnthropic(model=self.anthropic_model, api_key=self.anthropic_api_key)
    #     return None
    
    def __init__(self):
        self.project_id = os.getenv("PROJECT_ID")
        assert self.project_id, "Project ID is required"
        self.location = os.getenv("LOCATION")
        assert self.location, "Location is required"
        self.gemini_model = os.getenv("GEMINI_MODEL")
        assert self.gemini_model, "Gemini model is required"
        self.gemma_endpoint_id = os.getenv("GEMMA_ENDPOINT_ID", "gemma-model-endpoint")
        self.hf_access_token = os.getenv("HF_ACCESS_TOKEN")
        assert self.hf_access_token, "Hugging Face access token is required"
    
    @property
    def vertexai_gemini(self):
        if self.project_id:
            return ChatVertexAI(model_name=self.gemini_model, 
                        project=self.project_id, 
                        location=self.location)
        return None
    
    @property
    def gemma_vertexai(self):
        if self.project_id:
            return GemmaVertexAIModelGarden(
                model_name="gemma-3-27b-it", 
                project=self.project_id, 
                location=self.location,
                temperature=0,
                endpoint_id=self.project_id)
        return None
        
    @property
    def gemma_hf(self):
        if self.hf_access_token:
            return GemmaLocalHF(
                model_name="google/gemma-3-1b-it", 
                hf_access_token=self.hf_access_token,
                max_tokens=1000,
                temperature=0)
        return None
    
    @property
    def gemma_chat_hf(self):
        if self.hf_access_token:
            return GemmaChatLocalHF(
                model_name="google/gemma-3-1b-it", 
                hf_access_token=self.hf_access_token,
                max_tokens=1000,
                temperature=0)
        return None

    @property
    def chat_mistral(self):
        api_key = os.getenv("MISTRAL_API_KEY")
        assert api_key, "Mistral API key is required"
        model = os.getenv("MISTRAL_MODEL", "mistral-large-latest")
        return ChatMistralAI(model=model, api_key=api_key)

    # @property
    # def google_vision(self):
    #     return vision.ImageAnnotatorClient()

cfg = Config()
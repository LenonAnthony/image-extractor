import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_vertexai import ChatVertexAI
from langchain_anthropic import ChatAnthropic
from langchain_mistralai import ChatMistralAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from google.cloud import vision
load_dotenv()

os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"
os.environ["GRPC_POLL_STRATEGY"] = "poll"

class Config:
    # quando for utilizar os modelos da openAI, tira os comentários abaixo de __init__ e chat_openai e
    #  comentar a parte do gemini (de project_id até o fim da função def vertexai, google vision ...etc)

    def __init__(self):
        # self.openai_api_key = os.getenv("OPENAI_API_KEY")
        # assert self.openai_api_key, "OpenAI API key is required"
        # self.openai_model = os.getenv("OPENAI_MODEL")
        # assert self.openai_model, "OpenAI model is required"  
    
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

    # @property
    # def google_vision(self):
    #     return vision.ImageAnnotatorClient()

    #     self.mistral_api_key = os.getenv("MISTRAL_API_KEY")
    #     assert self.mistral_api_key, "Mistral API key is required"
    #     self.mistral_model = os.getenv("MISTRAL_MODEL")
    #     assert self.mistral_model, "Mistral model is required"
        
    # @property
    # def chat_mistral(self):
    #     """Instancia `ChatMistralAI` apenas quando necessário"""
    #     return ChatMistralAI(model=self.mistral_model, api_key=self.mistral_api_key)

    #     self.ollama_model = os.getenv("OLLAMA_MODEL")
    #     assert self.ollama_model, "Ollama model is required"
        
    # @property
    # def chat_ollama(self):
    #     """Instancia `ChatOllama` apenas quando necessário"""
    #     return ChatOllama(model=self.ollama_model)

        self.huggingface_repo_id = os.getenv("HUGGINGFACE_REPO_ID")
        assert self.huggingface_repo_id, "HuggingFace repo id is required"
        self.huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")
        assert self.huggingface_api_key, "HuggingFace API key is required"
        
    @property
    def chat_huggingface(self):
        """Instancia `HuggingFaceEndpoint` apenas quando necessário"""
        llm = HuggingFaceEndpoint(repo_id=self.huggingface_repo_id,task="text-generation", max_new_tokens=512,huggingfacehub_api_token=self.huggingface_api_key)

        return ChatHuggingFace(llm=llm, verbose=True)
        


cfg = Config()
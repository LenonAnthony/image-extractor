import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_vertexai import ChatVertexAI
from google.cloud import vision
load_dotenv()

os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"
os.environ["GRPC_POLL_STRATEGY"] = "poll"

class Config:
    # quando for utilizar os modelos da openAI, tira os comentários abaixo e
    #  comenta a parte do gemini (de project_id até o fim da função def vertexai ...etc)

    openai_api_key = os.getenv("OPENAI_API_KEY")
    assert openai_api_key, "OpenAI API key is required"
    openai_model = os.getenv("OPENAI_MODEL")
    assert openai_model, "OpenAI model is required"
    chat_openai = ChatOpenAI(model=openai_model, api_key=openai_api_key)
    
    # project_id = os.getenv("PROJECT_ID")
    # assert project_id, "Project ID is required"
    # location = os.getenv("LOCATION")
    # assert location, "Location is required"
    # gemini_model = os.getenv("GEMINI_MODEL")
    # assert gemini_model, "Gemini model is required"
    
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
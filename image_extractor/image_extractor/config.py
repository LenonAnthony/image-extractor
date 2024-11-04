import os
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv

load_dotenv()


class Config:
    open_ai_key = os.getenv("OPENAI_API_KEY")
    assert open_ai_key is not None, "There is no Open AI key"
    open_ai_model = os.getenv("OPENAI_MODEL")
    assert open_ai_model is not None, "Please specify your OpenAI model"
    chat_open_ai = ChatOpenAI(model=open_ai_model, api_key=open_ai_key)

    gemini_api_key = os.getenv("GEMINI_API_KEY")
    assert gemini_api_key is not None, "Cannot find Gemini API key"
    google_model = os.getenv("GOOGLE_MODEL")
    assert google_model is not None, "Please specify your Google Gemini model"
    google_ai = ChatGoogleGenerativeAI(model=google_model, api_key=gemini_api_key)


cfg = Config()

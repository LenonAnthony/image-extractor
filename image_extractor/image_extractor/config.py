import os
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

load_dotenv()


class Config:
    open_ai_key = os.getenv("OPENAI_API_KEY")
    assert open_ai_key is not None, "There is no Open AI key"
    open_ai_model = os.getenv("OPENAI_MODEL")
    assert open_ai_model is not None, "Please specify your OpenAI model"
    chat_open_ai = ChatOpenAI(model=open_ai_model, api_key=open_ai_key)


cfg = Config()

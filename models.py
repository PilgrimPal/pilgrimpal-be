from pydantic import BaseModel


class ChatbotRes(BaseModel):
    response: str


class ChatbotReqBody(BaseModel):
    session_id: str
    prompt: str


from langchain.pydantic_v1 import BaseModel, Field


class TranslatorInput(BaseModel):
    text: str = Field(description="text to translate")
    lang_tgt: str = Field(description="target language")

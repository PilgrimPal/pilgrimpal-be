from pydantic import BaseModel


class ChatbotRes(BaseModel):
    response: str


class ChatbotReqBody(BaseModel):
    session_id: str
    prompt: str

from pydantic import BaseModel
import models


class ChatbotRes(BaseModel):
    title: str
    response: str


class ChatbotReqBody(BaseModel):
    session_id: str
    prompt: str


class ChatHistoryRes(BaseModel):
    title: str
    messages: list[models.MessageStore]


class ChatTitlesRes(BaseModel):
    titles: list[models.ChatTitle]

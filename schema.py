from pydantic import BaseModel
import models
from typing import Optional


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


class CrowdDetail(BaseModel):
    crowd_count: int
    crowd_density: float
    updated_at: str


class CrowdDetailRes(BaseModel):
    crowd_count: Optional[int]
    crowd_density: Optional[float]
    updated_at: Optional[str]
    avg_crowd_count: float
    avg_crowd_density: float
    crowd_history: list[CrowdDetail]

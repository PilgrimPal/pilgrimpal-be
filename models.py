from pydantic import BaseModel
from datetime import datetime
from uuid import UUID
from datetime import datetime


class ChatTitle(BaseModel):
    id: UUID
    created_at: datetime
    session_id: str
    title: str


class MessageStore(BaseModel):
    id: int
    session_id: str
    message: str

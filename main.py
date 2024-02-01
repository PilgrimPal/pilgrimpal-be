# Standard library imports
import asyncio
import json
import os

# FastAPI imports
from fastapi import FastAPI
from dotenv import load_dotenv
from contextlib import asynccontextmanager
import models

# Load environment variables from .env file
load_dotenv()

# OpenAI imports
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

# import chatbot engine
from chatbot import ChatbotEngine

# instance for chatbot engine
chatbot_engine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # before app startup
    # init chatbot engine
    global chatbot_engine
    chatbot_engine = ChatbotEngine(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        gsearch_key=os.getenv("GSEARCH_KEY"),
        gsearch_engine_id=os.getenv("GSEARCH_ENGINE_ID"),
        postgres_user=os.getenv("POSTGRES_USER"),
        postgres_password=os.getenv("POSTGRES_PASSWORD"),
        postgres_host=os.getenv("POSTGRES_HOST"),
        postgres_port=os.getenv("POSTGRES_PORT"),
        postgres_db=os.getenv("POSTGRES_DB"),
    )
    yield
    # after app shutdown


app = FastAPI(lifespan=lifespan)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/chat")
async def post_chat(body: models.ChatbotReqBody) -> models.ChatbotRes:
    response = chatbot_engine.chat(body.session_id, body.prompt)
    return models.ChatbotRes(response=response)

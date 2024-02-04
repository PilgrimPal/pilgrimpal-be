# Standard library imports
import asyncio
import json
import os

# FastAPI imports
from fastapi import FastAPI, APIRouter
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from databases import Database
import schema

# Load environment variables from .env file
load_dotenv()

# OpenAI imports
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

# import engine
from chatbot import ChatbotEngine
from crowd_counter import CrowdCounter

# define instances
chatbot_engine = None
crowd_counter_engine = None
db = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # before app startup
    # init chatbot engine
    global chatbot_engine, crowd_counter_engine, db
    db_url = f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
    db = Database(db_url)
    await db.connect()
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
    crowd_counter_engine = CrowdCounter()
    yield
    # after app shutdown
    await db.disconnect()


app = FastAPI(lifespan=lifespan)


@app.get("/")
def read_root():
    return {"Hello": "World"}


chatbot_router = APIRouter(tags=["chatbot"])


@chatbot_router.post("/chat")
async def post_chat(body: schema.ChatbotReqBody) -> schema.ChatbotRes:
    response = await chatbot_engine.chat(body.session_id, body.prompt)
    title = await db.fetch_one(
        "SELECT title FROM chat_title WHERE session_id = :session_id",
        {"session_id": body.session_id},
    )
    if not title:
        title = chatbot_engine.generate_title(body.prompt)
        await db.execute(
            "INSERT INTO chat_title (session_id, title) VALUES (:session_id, :title)",
            {"session_id": body.session_id, "title": title},
        )
    return {
        "title": title,
        "response": response,
    }


@chatbot_router.get("/chat_history/{session_id}")
async def get_chat_history(session_id: str) -> schema.ChatHistoryRes:
    messages = await db.fetch_all(
        "SELECT * FROM message_store WHERE session_id = :session_id",
        {"session_id": session_id},
    )
    chat_title = await db.fetch_one(
        "SELECT title FROM chat_title WHERE session_id = :session_id",
        {"session_id": session_id},
    )
    return {
        "title": chat_title.title,
        "messages": messages,
    }


@chatbot_router.get("/chat_titles")
async def get_chat_history() -> schema.ChatTitlesRes:
    chat_titles = await db.fetch_all(
        "SELECT * FROM chat_title ORDER BY created_at DESC"
    )
    return {
        "titles": chat_titles,
    }


app.include_router(chatbot_router, prefix="/api/chatbot")

crowd_router = APIRouter(tags=["crowd"])


@crowd_router.get("/crowd")
async def post_crowd():
    response = crowd_counter_engine.inference("./crowd_counter/vis/umroh.png")
    return response


app.include_router(crowd_router, prefix="/api/crowd")

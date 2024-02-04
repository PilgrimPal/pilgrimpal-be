# Standard library imports
import asyncio
import json

# import os
from datetime import datetime

# FastAPI imports
from fastapi import FastAPI, APIRouter, WebSocket, WebSocketDisconnect
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from databases import Database
import schema
from redis import Redis
from lib.socket_manager import PubSubWebSocketManager
from config.settings import get_settings

# Load environment variables from .env file
# load_dotenv()

# OpenAI imports
import openai

settings = get_settings()
openai.api_key = settings.OPENAI_API_KEY

# import engine
from chatbot import ChatbotEngine
from crowd_counter import CrowdCounter

# define instances
db = None
redis = None
psws_manager = None
chatbot_engine = None
crowd_counter_engine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # before app startup
    global db, redis, psws_manager, chatbot_engine, crowd_counter_engine

    # init database
    # db_url = f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
    db = Database(settings.DATABASE_URL)
    await db.connect()

    # init redis connection
    redis = Redis(
        db=0,
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        password=settings.REDIS_PASSWORD,
        decode_responses=True,
    )

    # init pubsub ws manager
    psws_manager = PubSubWebSocketManager(
        redis_host=settings.REDIS_HOST,
        redis_port=settings.REDIS_PORT,
        redis_password=settings.REDIS_PASSWORD,
    )

    # check redis connection
    redis.ping()

    # init chatbot engine
    chatbot_engine = ChatbotEngine()

    # init crowd counter engine
    crowd_counter_engine = CrowdCounter()

    # Start polling task
    asyncio.create_task(poll())

    yield

    # after app shutdown
    await psws_manager.close_subscribers()
    await db.disconnect()


async def poll():
    print("Poll started")
    i = 1
    while True:
        img_path = f"./images/area-1/img-{i}.jpg"

        response = crowd_counter_engine.inference(img_path)
        crowd_dict = {
            "crowd_count": response[0],
            "crowd_density": response[1],
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        await psws_manager.broadcast_to_channel("area.1", json.dumps(crowd_dict))

        if i == 35:
            i = 1
        await asyncio.sleep(30)


app = FastAPI(lifespan=lifespan)


@app.get("/")
def read_root():
    return {"Hello": "World"}


chatbot_router = APIRouter(tags=["chatbot"])


@chatbot_router.post("/chat")
async def execute_prompt(body: schema.ChatbotReqBody) -> schema.ChatbotRes:
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
async def get_chat_titles() -> schema.ChatTitlesRes:
    chat_titles = await db.fetch_all(
        "SELECT * FROM chat_title ORDER BY created_at DESC"
    )
    return {
        "titles": chat_titles,
    }


app.include_router(chatbot_router, prefix="/api/chatbot")

crowd_router = APIRouter(tags=["crowd"])


@crowd_router.get("/crowd")
async def get_crowd():
    response = crowd_counter_engine.inference("./crowd_counter/vis/umroh.png")
    return response


@crowd_router.websocket("/{area_id}/ws")
async def subscribe_bus_location(websocket: WebSocket, area_id: str) -> None:
    channel = f"area.{area_id}"
    await psws_manager.subscribe_to_channel(channel, websocket)

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await psws_manager.disconnect_from_channel(channel, websocket)


app.include_router(crowd_router, prefix="/api/crowd")

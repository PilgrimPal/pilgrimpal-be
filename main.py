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
        i = (i % 35) + 1

        response = crowd_counter_engine.inference(img_path)
        crowd_dict = {
            "crowd_count": response[0],
            "crowd_density": response[1],
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        await psws_manager.broadcast_to_channel("ps:area:1", json.dumps(crowd_dict))
        redis.lpush("list:area:1", json.dumps(crowd_dict))
        if redis.llen("list:area:1") > 60:  # 60 * 30s = 30 minutes
            redis.ltrim("list:area:1", 0, 4)

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
    else:
        title = title.title
    return {
        "title": title,
        "response": response,
    }


@chatbot_router.get("/chat_history/{session_id}")
async def get_chat_history(session_id: str):
    messages = await db.fetch_all(
        "SELECT * FROM message_store WHERE session_id = :session_id",
        {"session_id": session_id},
    )
    chat_title = await db.fetch_one(
        "SELECT title FROM chat_title WHERE session_id = :session_id",
        {"session_id": session_id},
    )
    result = []
    for message in messages:
        message_dict = json.loads(message["message"])["data"]
        if message_dict["type"] == "human":
            m = message_dict["content"].split("[<</LIM]\n\nUser Prompt: ")[1].replace(" [/INST]\n\n", "")
            message_dict["content"] = m
        result.append(message_dict)
    return {
        "title": chat_title.title,
        "messages": result,
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


@crowd_router.get("/areas")
async def get_crowd_areas():
    area_ids = redis.keys("list:area:*")
    response = {}
    for area_id in area_ids:
        area = area_id.split(":")[-1]
        crowd_history = redis.lrange(area_id, 0, 1)
        crowd_detail = json.loads(crowd_history[0]) if crowd_history else {}
        response[area] = crowd_detail["crowd_density"]
    return response


@crowd_router.get("/{area_id}")
async def get_crowd_detail(area_id: str) -> schema.CrowdDetailRes:
    crowd_history = redis.lrange(f"list:area:{area_id}", 0, -1)
    if crowd_history:
        crowd_history = [json.loads(crowd) for crowd in crowd_history]
    else:
        crowd_history = []
    latest_crowd = crowd_history[-1] if crowd_history else {}

    avg_count, avg_density, history_len = 0, 0, len(crowd_history)
    if history_len > 0:
        for crowd in crowd_history:
            avg_count += crowd.get("crowd_count", 0)
            avg_density += crowd.get("crowd_density", 0)
        avg_count = avg_count / history_len
        avg_density = avg_density / history_len

    return {
        "crowd_count": latest_crowd.get("crowd_count", None),
        "crowd_density": latest_crowd.get("crowd_density", None),
        "updated_at": latest_crowd.get("updated_at", None),
        "avg_crowd_count": avg_count,
        "avg_crowd_density": avg_density,
        "crowd_history": crowd_history,
    }


app.include_router(crowd_router, prefix="/api/crowd")

ws_router = APIRouter(tags=["websocket"])


@ws_router.websocket("/{area_id}")
async def subscribe_bus_location(websocket: WebSocket, area_id: str) -> None:
    channel = f"ps:area:{area_id}"
    await psws_manager.subscribe_to_channel(channel, websocket)

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await psws_manager.disconnect_from_channel(channel, websocket)


app.include_router(ws_router, prefix="/ws/crowd")

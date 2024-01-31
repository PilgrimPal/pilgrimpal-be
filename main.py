# Standard library imports
import asyncio
import json
import os

# FastAPI imports
from fastapi import FastAPI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# OpenAI imports
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

# LlamaIndex imports
from llama_index.agent import OpenAIAgent
from llama_index.tools.tool_spec.load_and_search.base import LoadAndSearchToolSpec
from llama_index import VectorStoreIndex, download_loader
from llama_hub.tools.google_search.base import GoogleSearchToolSpec
from llama_hub.tools.azure_translate import AzureTranslateToolSpec
from llama_index.langchain_helpers.agents import (
    IndexToolConfig,
    LlamaIndexTool,
)

# LangChain imports
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import OpenAI, ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import ChatPromptTemplate
from langchain_community.utilities import GoogleSearchAPIWrapper


app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}

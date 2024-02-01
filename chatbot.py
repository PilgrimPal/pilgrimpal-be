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
from langchain.memory import PostgresChatMessageHistory
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import OpenAI, ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import ChatPromptTemplate
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain.tools import BaseTool, StructuredTool, tool


class ChatbotEngine:
    def __init__(
        self,
        openai_api_key: str,
        gsearch_key: str,
        gsearch_engine_id: str,
        postgres_user: str,
        postgres_password: str,
        postgres_host: str,
        postgres_port: str,
        postgres_db: str,
    ) -> None:
        self._openai_api_key = openai_api_key
        self._gsearch_key = gsearch_key
        self._gsearch_engine_id = gsearch_engine_id
        self._postgres_user = postgres_user
        self._postgres_password = postgres_password
        self._postgres_host = postgres_host
        self._postgres_port = postgres_port
        self._postgres_db = postgres_db

        # init tools
        self._tools = self._init_tools()

        # init model
        self._model = self._init_model()

        # init prompt template
        self._prompt_temp = self._init_prompt_temp()

    def _init_tools(self) -> list[Tool]:
        tools = []
        tools.append(self._get_gsearch_tool())
        def python(script):
            """
            Execute python code
            """
            try:
                # exec(script)
                return "Arbitrary python executable is not supported."
            except Exception as e:
                print(e)
                # sentry_sdk.capture_exception(e)
                return False


        python_tool = StructuredTool.from_function(
            func=python,
            name="Python",
            description="useful when you need to execute python script.",
            # coroutine= ... <- you can specify an async method if desired as well
        )
        tools.append(python_tool)
        return tools

    def _init_model(self) -> ChatOpenAI:
        return ChatOpenAI(
            openai_api_key=self._openai_api_key,
            temperature=0,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()],
        )

    def _init_prompt_temp(self) -> ChatPromptTemplate:
        # prompt template
        system_prompt = "You are a helpful Pilgrim AI Assistant and an expert in umroh and hajj, who provides precise, correct and concise answers."
        limit_prompt = "Before answering the user prompt, only answer if the prompt is related to umroh or hajj or language translation. \
        Otherwise don't use any tools and reject the question formally. \
        If the user prompt is about language translation, you are prohibited to use the 'Search Engine' tool. \
        Other than that, you are free to use the tool."
        user_prompt = "{input}"

        B_INST, E_INST = "[INST]", "[/INST]"
        B_LIM, E_LIM = "[LIM>>]", "[<</LIM]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

        template = f"{B_INST} {B_SYS}{system_prompt.strip()}{E_SYS}{B_LIM}Limit: {limit_prompt.strip()}{E_LIM}\n\nUser Prompt: {user_prompt} {E_INST}\n\n"

        return ChatPromptTemplate.from_template(template)

    def _get_gsearch_tool(self) -> Tool:
        google_spec = GoogleSearchToolSpec(
            key=self._gsearch_key,
            engine=self._gsearch_engine_id,
        )

        # Wrap the google search tool as it returns large payloads
        gsearch_tools = LoadAndSearchToolSpec.from_defaults(
            google_spec.to_tool_list()[0],
        ).to_tool_list()

        # create the agent
        gsearch_agent = OpenAIAgent.from_tools(gsearch_tools, verbose=False)

        # create the tool
        gsearch_tool_config = IndexToolConfig(
            query_engine=gsearch_agent,
            name=f"Search Engine",
            description=f"Useful when you want answer for the questions on Google.",
        )

        return LlamaIndexTool.from_tool_config(gsearch_tool_config)

    def chat(self, session_id: str, prompt: str) -> str:
        # define the message history
        connection_string = f"postgresql://{self._postgres_user}:{self._postgres_password}@{self._postgres_host}:{self._postgres_port}/{self._postgres_db}"
        message_history = PostgresChatMessageHistory(
            connection_string=connection_string,
            session_id=session_id,
        )

        # define the memory
        memory = ConversationBufferMemory(
            memory_key="chat_history", chat_memory=message_history, return_messages=True
        )

        # create the agent
        agent_chain = initialize_agent(
            self._tools,
            self._model,
            memory=memory,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True,
        )

        # chat
        input = self._prompt_temp.format_messages(
            input=prompt,
        )[0].content
        response = agent_chain.run(input)

        return response

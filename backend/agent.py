import os
import logging

from dotenv import load_dotenv
from typing import Literal

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain_community.tools import DuckDuckGoSearchRun

from vector_storage import vector_storage
from tools import get_current_time, get_weather, get_user_statistics

log = logging.getLogger(__name__)

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

internet_search = DuckDuckGoSearchRun(
    name="internet_search",
    description="Use this tool to search the internet for current events, news, or any information you don't know."
)

tools = [get_weather, get_current_time, get_user_statistics, internet_search]
model = ChatGroq(temperature=0.3, model="llama-3.3-70b-versatile")
model_with_tools = model.bind_tools(tools)

class AgentState(MessagesState):
    context: str
    user_id: str

async def call_model(state: AgentState):
    log.info(f"Agent Node: Answering User: {state.get('user_id')}")
    messages = state["messages"]
    context = state.get("context", "")

    system_message = {
        "role": "system",
        "content": (
            "You are an intelligent AI assistant. "
            "CRITICAL INSTRUCTION: You DO have access to the user's uploaded documents. "
            "The text extracted from their files is provided below in the CONTEXT section. "
            "NEVER say you cannot read or access files. If the user asks about their document, uploaded file, or PDF, ALWAYS assume the information is in the CONTEXT below and use it. "
            "If the information is not in the context, use your 'internet_search' tool to find it on the internet."
            f"Context from user's documents:{context}"
        )
    }

    response = await model_with_tools.ainvoke([system_message] + messages)
    if response.tool_calls:
        for tool in response.tool_calls:
            log.info(f"Agent DECISION: Calling tool '{tool['name']}' with arguments {tool['args']}")

    return {"messages": [response]}

async def retriever_node(state: AgentState):
    user_input = state["messages"][-1].content
    user_id = state.get("user_id")
    log.info(f"Retriever Node: Searching the database for User: {state.get('user_id')}")

    try:
        docs = vector_storage.similarity_search(
            user_input,
            k=10,
            filter={"user_id": user_id}
        )
        context = "\n\n".join(d.page_content for d in docs)
        log.info(f"Retriever SUCCESS: Found {len(docs)} splits.")
        return {"context": context}
    except Exception as e:
        log.error(f"Retriever ERROR: Error while searching the database: {str(e)}")
        return {"context": f"Context not found: {e}"}

tool_node = ToolNode(tools)

def should_continue(state: AgentState) -> Literal["tools", END]:
    last_message = state["messages"][-1]

    if last_message.tool_calls:
        log.info("Graph Edge: Calling Tools")
        return "tools"

    log.info("Graph Edge: Answering User.")
    return END

workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("retriever", retriever_node)
workflow.add_node("tools", tool_node)
workflow.add_edge(START, "retriever")
workflow.add_edge("retriever", "agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")

async def get_response(user_input: str, user_id: str, thread_id: str):
    log.info(f"Agent START: Recieved activity from User: {user_id} in Thread: {thread_id}")
    config = {"configurable": {"thread_id": thread_id}}

    async with AsyncSqliteSaver.from_conn_string("checkpoints.db") as memory:
        await memory.setup()
        agent_app = workflow.compile(checkpointer=memory)
        result = await agent_app.ainvoke({"messages": [("user", user_input)], "user_id": user_id}, config=config)

    return result["messages"][-1].content
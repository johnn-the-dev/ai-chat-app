import os
from dotenv import load_dotenv
import asyncio
import os
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

model = ChatGroq(temperature=0.3, model="openai/gpt-oss-120b")

async def call_model(state: MessagesState):
    response = await model.ainvoke(state["messages"])
    return {"messages": response}

workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.add_edge(START, "agent")
workflow.add_edge("agent", END)

memory = MemorySaver()
agent_app = workflow.compile(checkpointer=memory)

async def get_response(user_input: str, user_id: str):
    config = {"configurable": {"thread_id": user_id}}

    result = await agent_app.ainvoke(
        {"messages": [("user", user_input)]},
        config=config
    )
    return result["messages"][-1].content
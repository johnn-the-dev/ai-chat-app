import os
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
from vector_storage import vector_storage

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

model = ChatGroq(temperature=0.3, model="llama-3.3-70b-versatile")

class AgentState(MessagesState):
    context: str
    user_id: str

async def call_model(state: AgentState):
    messages = state["messages"]
    context = state.get("context", "")

    system_message = {
        "role": "system",
        "content": f"You're an assistent. Use this context to answer the user: {context}"
    }

    response = await model.ainvoke([system_message] + messages)
    return {"messages": response}

async def retriever_node(state: AgentState):
    user_input = state["messages"][-1].content
    user_id = state.get("user_id")
    docs = vector_storage.similarity_search(
        user_input,
        k=10,
        filter={"user_id": user_id}
    )
    context = "\n\n".join(d.page_content for d in docs)
    return {"context": context}

workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("retriever", retriever_node)
workflow.add_edge(START, "retriever")
workflow.add_edge("retriever", "agent")
workflow.add_edge("agent", END)

memory = MemorySaver()
agent_app = workflow.compile(checkpointer=memory)

async def get_response(user_input: str, user_id: str):
    config = {"configurable": {"thread_id": user_id}}

    result = await agent_app.ainvoke(
        {
            "messages": [("user", user_input)],
            "user_id": user_id
         },
        config=config,
    )

    return result["messages"][-1].content
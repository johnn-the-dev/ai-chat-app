from fastapi import FastAPI, Depends
from pydantic import BaseModel
from agent import get_response
import database
from sqlalchemy.orm import Session
import requests

database.Base.metadata.create_all(bind=database.engine)

app = FastAPI(title="My API Chat")

@app.get("/")
async def read_root():
    return {"message": "Server online"}

class ChatMessage(BaseModel):
    message: str
    user_id: str

@app.post("/chat")
async def chat(data: ChatMessage, db: Session = Depends(database.get_db)):
    ai_answer = await get_response(data.message, data.user_id)

    new_log = database.ChatHistory(
        thread_id = "1",
        user_message = data.message,
        ai_response = ai_answer
    )
    db.add(new_log)
    db.commit()
    db.refresh(new_log)


    return {
        "user_input": data.message,
        "ai_response": ai_answer,
        "db_id": new_log.id
    }

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from pydantic import BaseModel
import database
from sqlalchemy.orm import Session
import os
from datetime import datetime
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader, TextLoader
import shutil
from vector_storage import vector_storage
from agent import get_response

database.Base.metadata.create_all(bind=database.engine)
app = FastAPI(title="My API Chat")

@app.get("/")
async def read_root():
    return {"message": "Server online"}

class ChatMessage(BaseModel):
    message: str
    user_id: str

class ChatMessageResponse(BaseModel):
    user_message: str
    ai_response: str
    timestamp: datetime

    class Config:
        from_attributes = True

@app.post("/chat")
async def chat(data: ChatMessage, db: Session = Depends(database.get_db)):
    ai_answer = await get_response(data.message, data.user_id)

    new_log = database.ChatHistory(
        thread_id = data.user_id,
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

@app.get("/history/{user_id}", response_model=list[ChatMessageResponse])
async def get_chat_history(user_id: str, db: Session = Depends(database.get_db)):
    history = db.query(database.ChatHistory).filter(database.ChatHistory.thread_id == user_id).order_by(database.ChatHistory.id.asc()).all()
    if not history:
        raise HTTPException(status_code=404, detail="History not found.")

    return history

@app.delete("/history/{user_id}")
async def delete_chat_history(user_id: str, db: Session = Depends(database.get_db)):
    db.query(database.ChatHistory).filter(database.ChatHistory.thread_id == user_id).delete()
    db.commit()

    return {"message": f"History for user {user_id} has been deleted."}

@app.post("/upload/{user_id}")
async def upload_file(user_id: str, file: UploadFile = File()):
    temp_path = f"temp_{user_id}_{file.filename}"
    with open(temp_path, "wb") as tmp:
        shutil.copyfileobj(file.file, tmp)

    try:
        if file.filename.endswith(".pdf"):
            loader = PyMuPDFLoader(temp_path)
        elif file.filename.endswith(".docx"):
            loader = Docx2txtLoader(temp_path)
        elif file.filename.endswith(".txt"):
            loader = TextLoader(temp_path)
        else:
            raise HTTPException(status_code=400, detail="This file format is not supported.")
        
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)

        for split in splits:
            split.metadata["user_id"] = user_id
        
        vector_storage.add_documents(splits)

        return {
            "status": "success",
            "message": "File successfully saved."
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error when handling file: {e}.")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
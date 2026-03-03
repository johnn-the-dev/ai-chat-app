import os
import database
import shutil
import logging
from datetime import datetime

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from pydantic import BaseModel
from sqlalchemy.orm import Session

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader, TextLoader

from vector_storage import vector_storage
from agent import get_response

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

database.Base.metadata.create_all(bind=database.engine)
app = FastAPI(title="My API Chat")

@app.get("/")
async def read_root():
    return {"message": "Server online"}

class ChatMessage(BaseModel):
    user_id: str
    message: str

class ChatMessageResponse(BaseModel):
    user_message: str
    ai_response: str
    timestamp: datetime

    class Config:
        from_attributes = True

@app.post("/chat")
async def chat(data: ChatMessage, db: Session = Depends(database.get_db)):
    log.info(f"Chat request - User: {data.user_id}, Message: {data.message[:50]}...")
    try:
        ai_answer = await get_response(data.message, data.user_id)

        new_log = database.ChatHistory(
            thread_id = data.user_id,
            user_message = data.message,
            ai_response = ai_answer
        )
        db.add(new_log)
        db.commit()
        db.refresh(new_log)

        log.info(f"Chat SUCCESS - Thread: {data.user_id}, DB_ID: {new_log.id}")
        return {
            "user_input": data.message,
            "ai_response": ai_answer,
            "db_id": new_log.id
        }
    except Exception as e:
        log.error(f"Chat FAILED - User: {data.user_id}, Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error.")
    

@app.get("/history/{user_id}", response_model=list[ChatMessageResponse])
async def get_chat_history(user_id: str, db: Session = Depends(database.get_db)):
    log.info(f"Chat History Fetch - User: {user_id}")
    history = db.query(database.ChatHistory).filter(database.ChatHistory.thread_id == user_id).order_by(database.ChatHistory.id.asc()).all()
    if not history:
        log.warning(f"Chat History EMPTY - User: {user_id}")
        raise HTTPException(status_code=404, detail="History not found.")

    log.info(f"Chat History Fetch SUCCESS - User: {user_id}, Entries: {len(history)}")
    return history

@app.delete("/history/{user_id}")
async def delete_chat_history(user_id: str, db: Session = Depends(database.get_db)):
    log.info(f"Chat History Delete - User: {user_id}")
    try:
        db.query(database.ChatHistory).filter(database.ChatHistory.thread_id == user_id).delete()
        db.commit()
        log.info(f"Chat History Delete SUCCESS - User: {user_id}")
        return {"message": f"History for user {user_id} has been deleted."}
    
    except Exception as e:
        log.error(f"Chat History Delete ERROR - User: {user_id}, Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Could not delete history.")

@app.post("/upload/{user_id}")
async def upload_file(user_id: str, file: UploadFile = File()):
    log.info(f"Upload START - User: {user_id}, File: {file.filename}")
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

        log.info(f"Upload SUCCESS - User: {user_id}, File: {file.filename}, Chunks: {len(splits)}")
        return {
            "status": "success",
            "message": "File successfully saved."
        }
    
    except Exception as e:
        log.error(f"Upload FAILED - User: {user_id}, File: {file.filename}, Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error when handling file: {e}.")

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/documents/{user_id}")
async def list_documents(user_id: str):
    log.info(f"Document list request - User: {user_id}")
    try:
        data = vector_storage.get(where={"user_id": user_id})
        if not data or not data["metadatas"]:
            log.info(f"Document list EMPTY - User: {user_id}")
            return {"user_id": user_id, "documents": []}

        files = set()
        for meta in data["metadatas"]:
            source = meta.get("source")
            if source:
                filename = os.path.basename(source)
                clean_name = filename.replace(f"temp_{user_id}_", "")
                files.add(clean_name)
        
        log.info(f"Document list SUCCESS - User: {user_id}, Found: {len(files)}")
        return {
            "user_id": user_id,
            "documents": list(files)
        }
    
    except Exception as e:
        log.error(f"Document list ERROR - User: {user_id}, Error: {e}")
        raise HTTPException(status_code=500, detail=f"Error while listing files: {str(e)}")
    
@app.delete("/documents/{user_id}/{filename}")
async def delete_file(user_id: str, filename: str):
    log.info(f"Document delete REQUEST - User: {user_id}, File: {filename}")
    try:
        db_name = f"temp_{user_id}_{filename}"
        data = vector_storage.get(where={
                "$and": [
                    {"user_id": {"$eq": user_id}},
                    {"source": {"$eq": db_name}}
                ]
            }
        )

        if not data or not data["ids"]:
            log.warning(f"Document delete NOT FOUND - User: {user_id}, File: {filename}")
            return {"message": "File not in database."}
        else:
            vector_storage.delete(where={
                    "$and": [
                        {"user_id": {"$eq": user_id}},
                        {"source": {"$eq": db_name}}
                    ]
                }
            )
            log.info(f"Document delete SUCCESS - User: {user_id}, File: {filename}, IDs removed: {len(data['ids'])}")
            return {
                "status": "success",
                "message": "File successfully deleted."
            }

    except Exception as e:
        log.error(f"Document delete ERROR - User: {user_id}, File: {filename}, Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error while deleting file: {e}")
    

    
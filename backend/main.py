import os
import database
import shutil
import logging
import random
import jwt
import bcrypt

from dotenv import load_dotenv
from agent import get_response
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from sqlalchemy.orm import Session

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader, TextLoader

from vector_storage import vector_storage
from fastapi.middleware.cors import CORSMiddleware

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM")
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", random.randint(1000, 2000)))

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(database.get_db)):
    login_information_error = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=JWT_ALGORITHM)
        username: str = payload.get("sub")
        if username is None:
            raise login_information_error
    
    except jwt.PyJWTError:
        raise login_information_error
    
    user = db.query(database.User).filter(database.User.username == username).first()
    if user is None:
        raise login_information_error
    return user

def verify_password(plain_password, hashed_password):
    password_byte_enc = plain_password.encode('utf-8')
    hashed_password_bytes = hashed_password.encode('utf-8')

    return bcrypt.checkpw(password_byte_enc, hashed_password_bytes)

def get_password_hash(password):
    pwd_bytes = password.encode('utf-8')
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(pwd_bytes, salt)
    return hashed_password.decode('utf-8')

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=JWT_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt

class UserCreate(BaseModel):
    username: str
    password: str

@app.post("/register")
async def register_user(user: UserCreate, db: Session = Depends(database.get_db)):
    db_user = db.query(database.User).filter(database.User.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already in use.")

    hashed_password = get_password_hash(user.password)
    new_user = database.User(username=user.username, hashed_password=hashed_password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"message": "User successfully registered."}

@app.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(database.get_db)):
    user = db.query(database.User).filter(database.User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect username or password.")
    
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer", "username": user.username}

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
async def chat(data: ChatMessage, db: Session = Depends(database.get_db), current_user: database.User = Depends(get_current_user)):
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
async def get_chat_history(user_id: str, db: Session = Depends(database.get_db), current_user: database.User = Depends(get_current_user)):
    log.info(f"Chat History Fetch - User: {user_id}")
    history = db.query(database.ChatHistory).filter(database.ChatHistory.thread_id == user_id).order_by(database.ChatHistory.id.asc()).all()
    if not history:
        log.warning(f"Chat History EMPTY - User: {user_id}")
        raise HTTPException(status_code=404, detail="History not found.")

    log.info(f"Chat History Fetch SUCCESS - User: {user_id}, Entries: {len(history)}")
    return history

@app.delete("/history/{user_id}")
async def delete_chat_history(user_id: str, db: Session = Depends(database.get_db), current_user: database.User = Depends(get_current_user)):
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
async def upload_file(user_id: str, file: UploadFile = File(), current_user: database.User = Depends(get_current_user)):
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
async def list_documents(user_id: str, current_user: database.User = Depends(get_current_user)):
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
async def delete_file(user_id: str, filename: str, current_user: database.User = Depends(get_current_user)):
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
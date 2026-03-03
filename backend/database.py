import logging
from datetime import datetime, timezone

from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

log = logging.getLogger(__name__)

SQLALCHEMY_DATABASE_URL = "sqlite:///./sql_app.db"

try:
    log.info(f"Database: Initializing connection to {SQLALCHEMY_DATABASE_URL}")
    engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    log.info(f"Database SUCCESS: Engine and SessionLocal successfully created.")
except Exception as e:
    log.error(f"Database ERROR: Failed to initialize the database: {str(e)}")

Base = declarative_base()

class ChatHistory(Base):
    __tablename__ = "chat_history"
    id = Column(Integer, primary_key=True, index=True)
    thread_id = Column(String)
    user_message = Column(String)
    ai_response = Column(String)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))

def get_db():
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        log.error(f"Database Session ERROR: {str(e)}")
        raise
    finally:
        db.close()
import logging

from sqlalchemy import func
from database import SessionLocal, ChatHistory
from langchain_core.tools import tool

log = logging.getLogger(__name__)

@tool
def get_user_statistics(user_id: str):
    """Returns the amount of messages sent and activity of user.
    """
    log.info(f"Tool CALL: 'get_user_statistics' for User: {user_id}")
    db = SessionLocal()
    try:
        count = db.query(func.count(ChatHistory.id)).filter(ChatHistory.thread_id == user_id).scalar()
        last_msg = db.query(ChatHistory.timestamp).filter(ChatHistory.thread_id == user_id).order_by(ChatHistory.id.desc()).first()

        if count == 0:
            result = f"User {user_id} hasn't sent a message yet."
            log.info(f"Tool SUCCESS: {result} ")
            return result

        timestamp_str = last_msg[0].strftime('%d.%m.%Y %H:%M')
        result = f"User has {count} messages. Last activity: {timestamp_str}."
        
        log.info(f"Tool SUCCESS: User {user_id} found with {count} messages.")
        return result

    except Exception as e:
        log.error(f"Tool ERROR: 'get_user_statistics' failed for User: {user_id}. Error: {str(e)}")
        return f"Error while handling statistics: {e}"
    finally:
        db.close()
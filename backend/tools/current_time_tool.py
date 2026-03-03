import logging

from datetime import datetime
from zoneinfo import ZoneInfo

from langchain_core.tools import tool

log = logging.getLogger(__name__)

@tool
def get_current_time(user_id: str, timezone_str: str):
    """Use this tool to get the current time in a specified timezone.
        The input should be a string like "UTC", "America/Los_Angeles", "Europe/Berlin".
        If not specified, use UTC.
    """
    log.info(f"Tool CALL: 'get_current_time' for User: {user_id}, Timezone: {timezone_str}.")
    try:
        tz = ZoneInfo(timezone_str.strip())
        now = datetime.now(tz)
        log.info(f"Tool SUCCESS: time information found for timezone: {timezone_str}.")
        return now.strftime("%d-%m-%Y %H-%M-%S %Z")

    except Exception as e:
        log.error(f"Tool Error: timezone '{timezone_str}' not found. Error: {str(e)}")
        return f"Error: Could not find timezone '{timezone_str}'. Use 'UTC' or 'America/New_York'."


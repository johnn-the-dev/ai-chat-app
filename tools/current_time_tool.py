from datetime import datetime
from zoneinfo import ZoneInfo
from langchain_core.tools import tool

@tool
def get_current_time(timezone_str: str):
    """Use this tool to get the current time in a specified timezone.
        The input should be a string like "UTS", "America/Los_Angeles", "Europe/Berlin".
    """

    try:
        tz = ZoneInfo(timezone_str.strip())
        now = datetime.now(tz)
        return now.strftime("%d-%m-%Y %H-%M-%S %Z")

    except Exception as e:
        return f"Error: Could not find timezone '{timezone_str}'. Use 'UTC' or 'America/New_York'."


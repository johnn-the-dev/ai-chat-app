import os
import requests
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()
API_KEY = os.getenv("API_KEY")

@tool
def get_weather(city_name: str, measurement: str = "metric") -> str:
    """
    Fetch current weather for a given city.
    'city_name' should be the name of the city (e.g., 'Prague').
    'measurement' should be 'metric' (Celsius, m/s) or 'imperial' (Fahrenheit, mph).
    """
    if not API_KEY:
        return "Error: OpenWeatherMap API key not found in environment variables."

    url = f"https://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={API_KEY}&units={measurement}"
    
    try:
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            
            if measurement == "imperial":
                temp_unit = "°F"
                wind_unit = "mph"

            else:
                temp_unit = "°C"
                wind_unit = "m/s"

            temp = data["main"]["temp"]
            feels_like = data["main"]["feels_like"]
            humidity = data["main"]["humidity"]
            wind_speed = data["wind"]["speed"]
            description = data["weather"][0]["description"]

            return (
                f"Weather in {city_name.capitalize()}: "
                f"Temperature: {temp}{temp_unit}, "
                f"Feels like: {feels_like}{temp_unit}, "
                f"Description: {description}, "
                f"Wind speed: {wind_speed} {wind_unit}, "
                f"Humidity: {humidity}%."
            )

        elif response.status_code == 404:
            return f"Error: City '{city_name}' not found."
        else:
            return f"Error: API returned status code {response.status_code}."

    except Exception as e:
        return f"Error connecting to weather service: {e}"
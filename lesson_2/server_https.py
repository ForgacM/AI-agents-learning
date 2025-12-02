from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP('weather')

api_base = 'https://api.open-meteo.com/v1'
user_agent = 'weather-app/1.0'

@mcp.tool()
async def get_current_temperature(latitude: float, longitude: float) -> str:
    """Get current temperature for a location.
      Args:
        latitude: Latitude of the location
        longitude: Longitude of the location

    Returns:
        str: JSON string with current weather data
    """
    url = f"{api_base}/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code"
    data = await make_openmeteo_request(url)

    if not data:
        return "Unable to fetch weather data."

    temp = data.get("current", {}).get("temperature_2m")
    return f"Current temperature: {temp}°C"

@mcp.tool()
async def get_current_weather(latitude: float, longitude: float) -> dict[str, Any]:
    """Get current weather for a location.
      Args:
        latitude: Latitude of the location
        longitude: Longitude of the location

    Returns:
        str: JSON string with current weather data
    """
    url = f"{api_base}/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code"
    data = await make_openmeteo_request(url)

    if not data:
        return {
            "success": False,
            "error": {"type": "DataError", "message": "No weather data received."}
        }

    return data.get("current", {})

@mcp.tool()
async def get_forecast(latitude: float, longitude: float) -> str:
    """Get weather forecast for a location.
    Args:
        latitude: Latitude of the location
        longitude: Longitude of the location

    Returns:
        str: Formatted weather forecast
    """
    url = f"{api_base}/forecast?latitude={latitude}&longitude={longitude}&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code&timezone=auto"
    data = await make_openmeteo_request(url)

    if not data:
        return "Unable to fetch forecast data."

    # Format the data for readability
    daily = data.get("daily", {})
    forecasts = []

    for i in range(len(daily.get("time", []))):
        forecast = f"""Date: {daily['time'][i]}
Max Temperature: {daily['temperature_2m_max'][i]}°C
Min Temperature: {daily['temperature_2m_min'][i]}°C
Precipitation: {daily['precipitation_sum'][i]} mm"""
        forecasts.append(forecast)

    return "\n---\n".join(forecasts)


async def make_openmeteo_request(url: str) -> dict[str, Any] | None:
    """Make request with proper error handling."""

    print("making request")

    headers = {
        "User-Agent": user_agent,
        "Accept": "application/json"
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        print(f"HTTP error {e.response.status_code}")
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    print("Starting server...")
    mcp.settings.host = "0.0.0.0"
    mcp.settings.port = 8002
    mcp.run(transport="streamable-http")
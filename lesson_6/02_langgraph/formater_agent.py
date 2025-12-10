import json
import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

load_dotenv()


class CityWeather(BaseModel):
    """Weather data for a single city."""
    city: str = Field(..., description="Name of the city")
    temperature: float = Field(default=0.0, description="Current temperature in Celsius")
    weather: str = Field(default="Unknown", description="Current weather description")
    date: str = Field(default="", description="Date of the weather (YYYY-MM-DD)")


class WeatherReport(BaseModel):
    """Complete weather report for multiple cities."""
    cities: list[CityWeather] = Field(..., description="Weather data for all requested cities")


async def create_formater_agent():
    """Create a formatter that returns structured JSON output."""

    llm = ChatAnthropic(
        model_name="claude-3-5-haiku-20241022",
        api_key=os.getenv("ANTHROPIC_API_KEY")
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a weather data formatter. Extract weather information and return structured data.

For EACH city you MUST include:
- city: the city name (string)
- temperature: current temperature in Celsius (number)
- weather: current weather description (string)
- date: the date of the weather data (YYYY-MM-DD format)

IMPORTANT: Every city object MUST have city, temperature, weather, and date fields."""),
        ("user", "{input}")
    ])

    chain = prompt | llm.with_structured_output(WeatherReport)

    return chain


def format_weather_json(weather_report: WeatherReport) -> str:
    """Convert WeatherReport to JSON string."""
    return json.dumps(weather_report.model_dump(), indent=2, ensure_ascii=False)

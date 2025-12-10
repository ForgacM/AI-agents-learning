import operator
import os
from datetime import datetime, timedelta
from typing import Annotated, Literal

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from .db_agent import create_db_agent
from .formater_agent import create_formater_agent, format_weather_json
from .weather_agent import create_weather_agent

load_dotenv()


def get_today_date() -> str:
    """Get today's date in YYYY-MM-DD format."""
    return datetime.now().strftime("%Y-%m-%d")


# ----------------------------------
# State
# ----------------------------------
class MyState(TypedDict):
    messages: Annotated[list, operator.add]
    # Cities to process
    cities: list[str]
    # Requested weather date
    weather_date: str
    # Cities found in cache
    cached_cities: list[str]
    # Cities needing weather fetch
    missing_cities: list[str]
    # Flag for conditional routing
    needs_fetch: bool


# ----------------------------------
# Pydantic model for extraction
# ----------------------------------
class ExtractedInfo(BaseModel):
    """Extracted cities and date from user message."""
    cities: list[str] = Field(description="List of city names extracted from the user message")
    weather_date: str = Field(description="The requested date in YYYY-MM-DD format. Use today's date if not specified.")


# ----------------------------------
# Node 0: Extract Cities from user message
# ----------------------------------
async def ExtractCitiesNode(state: MyState):
    """Extract city names and date from the user message using LLM."""
    print("----- Extract Cities Node -----")

    messages = state.get("messages", [])
    if not messages:
        return {"cities": [], "weather_date": get_today_date()}

    # Get the user message
    user_message = None
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "user":
            user_message = msg.get("content", "")
            break

    if not user_message:
        return {"cities": [], "weather_date": get_today_date()}

    print(f"   User message: {user_message}")

    today = get_today_date()
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

    # Use LLM with structured output to extract cities and date
    llm = ChatAnthropic(
        model_name="claude-3-5-haiku-20241022",
        api_key=os.getenv("ANTHROPIC_API_KEY")
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""Extract city names and the requested date from the user message.

Today's date is: {today}
Tomorrow's date is: {tomorrow}

For cities:
- Extract all city names mentioned
- If the user mentions a country or region (like 'Slovakia'), list the major cities

For date:
- If user says "today" or no date specified, use: {today}
- If user says "tomorrow", use: {tomorrow}
- If user specifies a date, convert it to YYYY-MM-DD format
- If user says "yesterday", calculate the date accordingly"""),
        ("user", "{message}")
    ])

    chain = prompt | llm.with_structured_output(ExtractedInfo)
    result = await chain.ainvoke({"message": user_message})

    cities = result.cities if result else []
    weather_date = result.weather_date if result else today
    print(f"   Extracted cities: {cities}")
    print(f"   Extracted date: {weather_date}")

    return {
        "cities": cities,
        "weather_date": weather_date,
    }


# ----------------------------------
# Node 1: Check Cache (DB)
# ----------------------------------
async def CheckCacheNode(state: MyState):
    """Check DB cache for existing weather data."""
    print("----- Check Cache Node -----")
    print(f"   Cities to check: {state.get('cities', [])}")

    # Invoke the DB agent to check cache
    db_agent = await create_db_agent()

    cities = state.get("cities", [])
    weather_date = state.get("weather_date", get_today_date())
    print(f"   Weather date: {weather_date}")

    # Build a message asking to check cache with clear instructions
    check_message = {
        "role": "user",
        "content": f"""Check cache for these cities: {cities}
For date: {weather_date}

For each city, call get_cached_temperature tool with weather_date="{weather_date}".
After checking ALL cities, respond with EXACTLY this format:
CACHED: city1, city2
MISSING: city3, city4

If no cities are cached, respond: CACHED: none
If all cities are cached, respond: MISSING: none"""
    }

    result = await db_agent.ainvoke({"messages": [check_message]})
    last_message = result["messages"][-1]

    response_content = str(last_message.content) if hasattr(last_message, 'content') else str(last_message)
    response_lower = response_content.lower()

    print("Response content:", response_content)

    cached = []
    missing = []

    # Check for negative indicators that suggest no cache
    no_cache_indicators = [
        "none of the cities",
        "no cached",
        "not found",
        "cache is empty",
        "no data",
        "don't have",
        "do not have",
        "haven't",
        "have not",
        "missing",
        "CACHED: none",
        "cached: none",
    ]

    has_no_cache = any(indicator in response_lower for indicator in no_cache_indicators)

    # Check for positive indicators that suggest data was found
    found_indicators = ["found", "cached data", "temperature", "°c", "celsius", "valid"]
    has_cache = any(indicator in response_lower for indicator in found_indicators) and not has_no_cache

    if has_no_cache or not has_cache:
        # No cache found - all cities need fetching
        missing = cities
        cached = []
    else:
        # Some data found - try to parse which cities have cache
        # Default to all cached if we can't parse
        cached = cities
        missing = []

    print(f"   Cached cities: {cached}")
    print(f"   Missing cities: {missing}")

    return {
        "messages": [last_message],
        "cached_cities": cached,
        "missing_cities": missing,
        "needs_fetch": len(missing) > 0,
    }


# ----------------------------------
# Node 2: Fetch Weather
# ----------------------------------
async def FetchWeatherNode(state: MyState):
    """Fetch weather for cities not in cache."""
    print("----- Fetch Weather Node -----")
    missing = state.get("missing_cities", [])
    weather_date = state.get("weather_date", get_today_date())
    print(f"   Fetching weather for: {missing}")
    print(f"   Weather date: {weather_date}")

    if not missing:
        return {"messages": [{"role": "assistant", "content": "No cities to fetch."}]}

    # Invoke the weather agent
    weather_agent = await create_weather_agent()

    # Build detailed message with city coordinates hints
    cities_info = "\n".join([f"- {city}" for city in missing])
    fetch_message = {
        "role": "user",
        "content": f"""Get current weather for these cities using your weather tools:
{cities_info}

The requested weather date is: {weather_date}

You MUST call the weather tool for EACH city. Use the city's coordinates (latitude, longitude).
Return the weather data you receive from the tools. The date for this weather data is: {weather_date}"""
    }

    result = await weather_agent.ainvoke({"messages": [fetch_message]})
    last_message = result["messages"][-1]

    print(f"   Weather data fetched")

    return {
        "messages": [last_message],
    }


# ----------------------------------
# Node 3: Store in Cache
# ----------------------------------
async def StoreCacheNode(state: MyState):
    """Store fetched weather data in cache."""
    print("----- Store Cache Node -----")

    # Get the last weather message to store
    messages = state.get("messages", [])
    if not messages:
        return {"messages": []}

    db_agent = await create_db_agent()

    # Get the weather data from last message
    last_msg = messages[-1]
    weather_content = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)

    weather_date = state.get("weather_date", get_today_date())
    cities = state.get("missing_cities", [])
    store_message = {
        "role": "user",
        "content": f"""Store this weather data in the database cache.

Cities to store: {cities}
Weather date: {weather_date}
Weather data: {weather_content}

You MUST call the store_temperature tool for EACH city with:
- the weather data (temperature, weather description)
- weather_date: {weather_date}
Confirm each city was stored."""
    }

    result = await db_agent.ainvoke({"messages": [store_message]})
    last_message = result["messages"][-1]

    print("   Data stored in cache")

    return {
        "messages": [last_message],
    }


# ----------------------------------
# Node 4: Compile Results
# ----------------------------------
async def CompileResultsNode(state: MyState):
    """Compile all results (cached + fetched) into final report."""
    print("----- Compile Results Node -----")

    formatter_chain = await create_formater_agent()

    # Extract weather data content from previous messages for context
    raw_messages = state.get("messages", [])
    weather_context = []

    for msg in raw_messages:
        # Handle AIMessage objects
        if hasattr(msg, 'content'):
            weather_context.append(str(msg.content))
        # Handle dict messages (skip system messages to avoid conflicts)
        elif isinstance(msg, dict) and msg.get("role") != "system":
            weather_context.append(str(msg.get("content", "")))

    weather_date = state.get("weather_date", get_today_date())

    # Build input for the formatter chain
    input_text = f"""Extract weather data for all cities from the following information.

IMPORTANT: The weather date is {weather_date}. Use this date for all weather data.

Weather data:
{chr(10).join(weather_context)}"""

    # Invoke the chain with structured output
    weather_report = await formatter_chain.ainvoke({"input": input_text})

    # Convert to JSON string
    json_output = format_weather_json(weather_report)

    return {
        "messages": [{"role": "assistant", "content": json_output}],
    }


# ----------------------------------
# Conditional Router: needs_fetch?
# ----------------------------------
def needs_fetch_router(state: MyState) -> Literal["fetch_weather", "compile_results"]:
    """Route based on whether we need to fetch missing cities."""
    needs_fetch = state.get("needs_fetch", False)
    print(f"----- Needs Fetch Router: {needs_fetch} -----")

    if needs_fetch:
        return "fetch_weather"
    else:
        return "compile_results"
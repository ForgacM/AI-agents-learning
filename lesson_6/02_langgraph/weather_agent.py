import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

async def get_mcp_tools():
    client = MultiServerMCPClient(
        {
            "weather": {
                "url": "http://localhost:8002/mcp",
                "transport": "streamable_http",
            }
        }
    )
    tools = await client.get_tools()
    return tools, client


async def create_weather_agent():

    mcp_tools, mcp_client = await get_mcp_tools()

    # model
    llm = ChatAnthropic(model_name="claude-3-5-haiku-20241022",
                        api_key=os.getenv("ANTHROPIC_API_KEY"))

    task = """You are a meteorologist with powerful tools to get current weather.

IMPORTANT: You MUST use your tools to get real weather data. Do not make up weather information.
When asked for weather data, ALWAYS call the appropriate weather tool with the city coordinates.

For each city requested:
1. Use the get_current_weather tool with latitude and longitude
2. Return the current temperature and weather description from the tool response"""

    # agent
    weather_agent = create_agent(
        llm,
        tools=mcp_tools,
        system_prompt=task,
    )

    return weather_agent
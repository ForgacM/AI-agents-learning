import asyncio
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_mcp_adapters.client import MultiServerMCPClient
from pydantic import BaseModel, Field

load_dotenv()

class ForecastResponse(BaseModel):
    """A movie with details."""

    date: str = Field(..., description="Date of event")
    temperature: int = Field(..., description="Temperature in Celsius.")
    weather: str = Field(..., description="Small weather description.")

class WeatherResponse(BaseModel):
    """A movie with details."""

    city: str = Field(..., description="Name of the city.")
    temperature: int = Field(..., description="Temperature in Celsius.")
    weather: str = Field(..., description="Small weather description.")
    forecast: list[ForecastResponse] = Field(..., description="The movie's rating out of 10")

async def main():

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

    mcp_tools, mcp_client = await get_mcp_tools()

    # model
    llm = ChatAnthropic(model_name="claude-3-5-haiku-20241022",
                        api_key=os.getenv("ANTHROPIC_API_KEY"))

    task = """You are a meteorologist with powerful tools to get weather and forecasts"""

    # agent
    agent = create_agent(
        llm,
        tools=mcp_tools,
        system_prompt=task,
    )

    conversation = {
        "messages": [
            {
                "role": "user",
                "content": "Check weather in biggest Slovak cities",
            },
        ]
    }

    # call model
    res = await agent.ainvoke(conversation)
    print("---- Response ----")

    structured_llm = llm.with_structured_output(list[WeatherResponse])

    result: WeatherResponse = await structured_llm.ainvoke(
        f"Convert this into WeatherResponse JSON:\n\n{res}"
    )
    print(result)

if __name__ == "__main__":
    asyncio.run(main())

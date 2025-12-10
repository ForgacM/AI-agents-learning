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
                "url": "http://localhost:8003/mcp",
                "transport": "streamable_http",
            }
        }
    )
    tools = await client.get_tools()
    return tools, client


async def create_db_agent():

    mcp_tools, mcp_client = await get_mcp_tools()

    # model
    llm = ChatAnthropic(model_name="claude-3-5-haiku-20241022",
                        api_key=os.getenv("ANTHROPIC_API_KEY"))

    task = """You are a database specialist with powerful tools for caching weather data.

IMPORTANT: You MUST use your tools. Do not make up responses.

When checking cache:
- Use the appropriate tool to check if cities exist in the cache
- Report which cities have cached data and which are missing

When storing data:
- Use the store_temperature or similar tool to save the weather data
- Confirm the data was stored successfully"""

    # agent
    db_agent = create_agent(
        llm,
        tools=mcp_tools,
        system_prompt=task,
    )

    return db_agent
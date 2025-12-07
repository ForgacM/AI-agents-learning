#!/usr/bin/env python3

import asyncio

from dotenv import load_dotenv

from .agents import ReActAgent

load_dotenv()


async def main():
    # Create agent with MCP tools
    agent = ReActAgent(
        name="Enhanced ReAct Assistant")

    try:
        # Connect to MCP server
        print("Connecting to MCP tools server...")
        await agent.connect()
        print("Connected successfully!\n")

        task = """Perform a comprehensive analysis:
        1. Check weather in biggest Slovak cities
        2. Get forecast for them for the next week
        4. Create a summary report in 'forecast_analysis.txt' with:
           - Highest temperatures in Slovak cities
           - Forecast table"""

        result = await agent.execute(task)
        print(f"Success: {result.success}")
        print(f"Result: {result.result}")
        print(f"Reasoning: {result.reasoning}")
        print(f"Actions taken: {result.actions_taken}")
        if result.error:
            print(f"Error: {result.error}")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Disconnect from MCP server
        await agent.disconnect()
        print("\nDisconnected from MCP server")


if __name__ == "__main__":
    asyncio.run(main())
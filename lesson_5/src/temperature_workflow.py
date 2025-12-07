#!/usr/bin/env python3
"""
Run the Temperature Cache Workflow.

This script demonstrates the cache-first pattern:
1. Check database for cached temperature
2. If not found or expired, fetch from weather API
3. Store result in database for future requests

The LLM determines city coordinates - no need to hardcode them.
"""

import asyncio
import sys
from dotenv import load_dotenv

from .agents import WorkflowAgent
from .workflows import MULTI_CITY_TEMPERATURE_WORKFLOW, EXAMPLE_CITIES

load_dotenv()


def _suppress_mcp_shutdown_errors(loop, context):
    """Suppress MCP library async cleanup errors during shutdown."""
    msg = context.get("message", "")
    exc = context.get("exception")

    # Suppress known MCP streamablehttp_client cleanup errors
    if exc and "cancel scope" in str(exc):
        return
    if "async_generator" in msg:
        return

    # For other exceptions, use default handler
    loop.default_exception_handler(context)

async def run_multi_city_workflow(cities: list[str] = None):
    """Run workflow for multiple cities."""
    if cities is None:
        cities = EXAMPLE_CITIES[:5]  # Default to first 5

    print("=" * 60)
    print("MULTI-CITY TEMPERATURE WORKFLOW")
    print("=" * 60)

    agent = WorkflowAgent(
        name="Multi-City Temperature Agent",
        model="claude-3.5-haiku",
    )

    try:
        await agent.connect()
        print("Connected to MCP servers\n")

        # Build the workflow
        agent.build_workflow(MULTI_CITY_TEMPERATURE_WORKFLOW)

        task = f"Get temperatures for these cities: {', '.join(cities)}. Use cache when available, fetch fresh data when needed."
        context = {"cities": cities}

        print(f"Task: {task}\n")

        result = await agent.execute(task, context)

        print("\n" + "=" * 60)
        print("RESULT")
        print("=" * 60)
        print(f"Success: {result.success}")
        print(f"Result: {result.result}")
        if result.error:
            print(f"Error: {result.error}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await agent.disconnect()
        print("\nDisconnected from MCP servers")


async def main():
    """Main entry point."""
    # Set custom exception handler to suppress MCP shutdown errors
    loop = asyncio.get_event_loop()
    loop.set_exception_handler(_suppress_mcp_shutdown_errors)

    await run_multi_city_workflow(EXAMPLE_CITIES)


if __name__ == "__main__":
    asyncio.run(main())

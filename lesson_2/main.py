import asyncio
import json
from typing import List, Dict, Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(
    base_url = "http://localhost:11434/v1/",  # Ollama’s local endpoint
    api_key = "ollama",  # required by the library but ignored by Ollama locally
)

server_parameter = StdioServerParameters(command="python", args=["./server.py"])

_EMPTY_SCHEMA = {"type": "object", "properties": {}}

def _to_json_schema(schema) -> dict:
    """Convert various schema formats to a JSON schema dict."""
    if schema is None:
        return _EMPTY_SCHEMA
    if hasattr(schema, "model_dump"):
        return schema.model_dump()
    if hasattr(schema, "dict"):
        return schema.dict()
    if isinstance(schema, str):
        try:
            return json.loads(schema)
        except (json.JSONDecodeError, TypeError):
            return _EMPTY_SCHEMA
    return schema

def _extract_tool_info(tool) -> tuple[str | None, str, dict]:
    """Extract name, description, and parameters from a tool (dict or object)."""
    if isinstance(tool, dict):
        return (
            tool.get("name"),
            tool.get("description", ""),
            _to_json_schema(tool.get("input_schema")),
        )
    return (
        getattr(tool, "name", None),
        getattr(tool, "description", "") or "",
        _to_json_schema(getattr(tool, "input_schema", None) or getattr(tool, "inputSchema", None)),
    )

def mcp_tools_to_openai(mcp_tools_response) -> list[dict]:
    """Convert MCP ToolsResponse to OpenAI function tools format."""
    tools_src = getattr(mcp_tools_response, "tools", mcp_tools_response) or []

    result = []
    for tool in tools_src:
        name, desc, params = _extract_tool_info(tool)
        if name:
            result.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": desc,
                    "parameters": params or _EMPTY_SCHEMA,
                },
            })
    return result

def mcp_tool_response_to_text(resp) -> str:
    """Pick structuredContent if present; else join text parts."""
    # Prefer structured JSON (dict)
    sc = getattr(resp, "structuredContent", None)
    if sc is not None:
        # Avoid double-encoding: sc is a dict already
        return json.dumps(sc, ensure_ascii=False)

    # Fall back to concatenating text parts
    parts = []
    for p in getattr(resp, "content", []) or []:
        # TextContent has .text; sometimes items are str already
        txt = getattr(p, "text", None)
        parts.append(txt if txt is not None else str(p))
    return "\n".join(parts) if parts else ""

class Agent:

    def __init__(self, model: str = "qwen3:4b"):
        self.model = model
        self.max_iterations = 10

    async def run(self, messages: List[Dict[str, Any]]) -> object:
        # Call the LLM
        async with stdio_client(server_parameter) as stdio_transport:
            async with ClientSession(*stdio_transport) as session:

                # 1) Handshake first
                init = await session.initialize()

                # 2) List tools AFTER initialize
                mcp_tools = await session.list_tools()

                # 3) Convert to OpenAI tool schema
                tools = mcp_tools_to_openai(mcp_tools.tools if hasattr(mcp_tools, "tools") else mcp_tools)

                iteration = 0
                while iteration < self.max_iterations:
                    iteration += 1
                    print(f"\n--- Iteration {iteration} ---")

                    # Call LLM
                    response = client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        tools=tools,
                        tool_choice="auto",
                    )

                    response_message = response.choices[0].message
                    print(f"LLM Response: {response_message}")

                    if response_message.tool_calls:
                        # Add the assistant's message to history
                        messages.append(response_message)

                        # Process ALL tool calls
                        for tool_call in response_message.tool_calls:
                            # Extract tool name and arguments
                            function_name = tool_call.function.name
                            function_args = json.loads(tool_call.function.arguments or "{}")

                            # Call the tool
                            tool_response = await session.call_tool(function_name, arguments=function_args)
                            print(f"Tool Response: {tool_response}")
                            # Convert MCP ToolResponse -> string for OpenAI tool message
                            tool_content = mcp_tool_response_to_text(tool_response)

                            messages.append({
                                "role": "tool",
                                "name": function_name,
                                "content": tool_content,
                            })
                        continue
                    else:
                        # Add the final assistant message to history
                        messages.append(response_message)
                        return response_message

                return "Something is wrong sorry no sorry"

def main():
    # Create a ReAct agent
    agent = Agent()

    print("=== Example 1: Single Tool Call ===")
    messages1 = [
        {"role": "system", "content": "You are a helpful AI assistant. "},
        {"role": "user", "content": "What is the current weather in Bratislava?"},
    ]

    result1 = asyncio.run(agent.run(messages1.copy()))
    print(f"\nResult: {result1.content}")

    print("=== Example 2: Multiple Tool Call ===")
    messages2 = [
        {"role": "system", "content": "You are a helpful AI assistant. "},
        {"role": "user", "content": "What is the current temperature in Bratislava? And how Forecast looks?"},
    ]

    result2 = asyncio.run(agent.run(messages2.copy()))
    print(f"\nResult: {result2.content}")

    print("=== Example 3: Reasoning ===")
    messages3 = [
        {"role": "system", "content": "You are a helpful AI assistant. "},
        {"role": "user", "content": "Compare temperatures in biggest Slovak cities and tell me where is warmest?"},
    ]

    result3 = asyncio.run(agent.run(messages3.copy()))
    print(f"\nResult: {result3.content}")

if __name__ == "__main__":
    main()
"""
Multi-MCP Client for connecting to multiple MCP servers.

This client aggregates tools from multiple MCP servers and routes
tool calls to the appropriate server.
"""

import logging
import os
from typing import List, Dict, Any, Optional

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

logger = logging.getLogger(__name__)


class MCPServerConnection:
    """Represents a connection to a single MCP server."""

    def __init__(self, name: str, server_url: str):
        self.name = name
        self.server_url = server_url.rstrip("/")
        self.session: Optional[ClientSession] = None
        self._http_context = None
        self._session_context = None
        self._streams = None
        self.tools: List[str] = []
        self._connected = False

    async def connect(self):
        """Connect to the MCP server."""
        try:
            # Enter the HTTP context
            self._http_context = streamablehttp_client(
                f"{self.server_url}/mcp", auth=None
            )
            self._streams = await self._http_context.__aenter__()
            read_stream, write_stream, _refresh = self._streams

            # Enter the session context
            self._session_context = ClientSession(read_stream, write_stream)
            self.session = await self._session_context.__aenter__()

            await self.session.initialize()

            # Cache tool names for routing
            response = await self.session.list_tools()
            self.tools = [tool.name for tool in response.tools]
            self._connected = True

            logger.info(f"Connected to MCP server '{self.name}' at {self.server_url}")
            logger.info(f"  Available tools: {self.tools}")
        except Exception as e:
            logger.error(f"Failed to connect to MCP server '{self.name}': {e}")
            self._connected = False
            raise

    async def disconnect(self):
        """Disconnect from the MCP server."""
        if not self._connected:
            return

        # Note: The MCP streamablehttp_client has async lifecycle issues
        # that cause errors on proper cleanup. We just mark as disconnected
        # and let the garbage collector clean up the connections.
        self.session = None
        self._session_context = None
        self._http_context = None
        self._streams = None
        self._connected = False

        logger.info(f"Disconnected from MCP server '{self.name}'")


class MultiMCPClient:
    """Client for connecting to multiple MCP servers."""

    def __init__(self, servers: Optional[Dict[str, str]] = None):
        """
        Initialize with server configuration.

        Args:
            servers: Dict mapping server names to URLs.
                     Example: {"weather": "http://localhost:8002", "db": "http://localhost:8003"}
                     If None, reads from environment variables.
        """
        if servers is None:
            servers = {
                "weather": os.getenv("MCP_WEATHER_URL", "http://localhost:8002"),
                "db": os.getenv("MCP_DB_URL", "http://localhost:8003"),
            }

        self.connections: Dict[str, MCPServerConnection] = {}
        for name, url in servers.items():
            self.connections[name] = MCPServerConnection(name, url)

        # Map tool names to server connections for routing
        self._tool_to_server: Dict[str, MCPServerConnection] = {}

    async def connect(self):
        """Connect to all MCP servers."""
        for name, conn in self.connections.items():
            try:
                await conn.connect()
                # Build tool routing map
                for tool_name in conn.tools:
                    self._tool_to_server[tool_name] = conn
            except Exception as e:
                logger.warning(f"Could not connect to server '{name}': {e}")

        if not self._tool_to_server:
            raise RuntimeError("Failed to connect to any MCP server")

        logger.info(f"Connected to {len([c for c in self.connections.values() if c._connected])} MCP server(s)")
        logger.info(f"Total tools available: {len(self._tool_to_server)}")

    async def disconnect(self):
        """Disconnect from all MCP servers."""
        for conn in self.connections.values():
            if conn._connected:
                try:
                    await conn.disconnect()
                except Exception as e:
                    logger.debug(f"Error disconnecting from '{conn.name}': {e}")

    async def list_tools(self):
        """List all available tools from all servers."""
        all_tools = []
        for conn in self.connections.values():
            if conn.session:
                response = await conn.session.list_tools()
                all_tools.extend(response.tools)
        return all_tools

    async def call_tool(self, tool_name: str, parameters: dict) -> str:
        """
        Call a tool, routing to the appropriate server.

        Args:
            tool_name: Name of the tool to call
            parameters: Parameters to pass to the tool

        Returns:
            Tool result as a string
        """
        # Find which server has this tool
        conn = self._tool_to_server.get(tool_name)
        if not conn:
            return f"Error: Tool '{tool_name}' not found on any connected server"

        if not conn.session:
            return f"Error: Server '{conn.name}' is not connected"

        try:
            response = await conn.session.call_tool(tool_name, parameters)

            # Extract text content from response
            if response.content:
                text_parts = []
                for content in response.content:
                    if hasattr(content, "text"):
                        text_parts.append(content.text)
                return "\n".join(text_parts)

            return "No response content"

        except Exception as e:
            error_msg = f"Error calling tool '{tool_name}' on server '{conn.name}': {str(e)}"
            logger.error(error_msg)
            return error_msg

    async def get_tools_definitions(self) -> List[Dict[str, Any]]:
        """Get OpenAI-compatible tool definitions from all MCP servers."""
        tools = await self.list_tools()
        openai_tools = []

        for tool in tools:
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema,
                },
            }
            openai_tools.append(openai_tool)

        return openai_tools

    def get_server_for_tool(self, tool_name: str) -> Optional[str]:
        """Get the server name that provides a specific tool."""
        conn = self._tool_to_server.get(tool_name)
        return conn.name if conn else None

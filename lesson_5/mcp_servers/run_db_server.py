#!/usr/bin/env python3
"""
Run the Database MCP Server with HTTP transport.

This script starts the database cache MCP server on HTTP.
Provides tools for caching temperature data in PostgreSQL.

Usage:
    python run_db_server.py [--port PORT]

Default port: 8003
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

# Load environment variables from the lesson_5 .env file
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
load_dotenv(env_path)

from db_server import mcp


def main():

    print(f"Database: {os.getenv('DB_HOST', 'localhost')}:{os.getenv('DB_PORT', '5433')}/{os.getenv('DB_NAME', 'weather_cache')}")
    print("Available tools:")
    print("  - get_cached_temperature (check cache)")
    print("  - store_temperature (store data)")
    print("  - get_all_cached_temperatures (list all)")
    print("  - delete_cached_temperature (remove city)")
    print("  - clear_expired_cache (cleanup)")
    print()

    print("Starting server...")
    mcp.settings.host = "0.0.0.0"
    mcp.settings.port = 8003
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()

"""
Database MCP Server for Temperature Cache (MySQL).

This server provides tools for storing and retrieving temperature data from MySQL.
It acts as a cache layer - check DB first before calling external weather APIs.

Usage:
    python db_server.py  # stdio transport
    or
    python run_db_server.py  # HTTP transport on port 8003
"""

import os
from datetime import datetime, timedelta
from typing import Any, Optional

import aiomysql
from mcp.server.fastmcp import FastMCP

print("Starting Database MCP Server (MySQL)...")

# Initialize FastMCP server
mcp = FastMCP('temperature-db')

# Database connection pool
_pool: Optional[aiomysql.Pool] = None


async def get_pool() -> aiomysql.Pool:
    """Get or create database connection pool."""
    global _pool
    if _pool is None:
        _pool = await aiomysql.create_pool(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "3306")),
            user=os.getenv("DB_USER", "weather_user"),
            password=os.getenv("DB_PASSWORD", "weather_password"),
            db=os.getenv("DB_NAME", "weather_cache"),
            minsize=1,
            maxsize=10,
            autocommit=True,
        )
        await init_db()
    return _pool


async def init_db():
    """Initialize database schema."""
    global _pool
    if _pool is None:
        return

    async with _pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute("""
                CREATE TABLE IF NOT EXISTS temperature_cache (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    city VARCHAR(255) NOT NULL,
                    latitude DECIMAL(10, 6) NOT NULL,
                    longitude DECIMAL(10, 6) NOT NULL,
                    temperature DECIMAL(5, 2) NOT NULL,
                    weather VARCHAR(255),
                    weather_date DATE NOT NULL,
                    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP DEFAULT (CURRENT_TIMESTAMP + INTERVAL 1 HOUR),
                    UNIQUE KEY unique_city_date (city, weather_date)
                )
            """)

            await cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_temperature_cache_city
                ON temperature_cache(city)
            """)

            await cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_temperature_cache_expires
                ON temperature_cache(expires_at)
            """)

            print("Database schema initialized.")


@mcp.tool()
async def get_cached_temperature(city: str, weather_date: str = None) -> dict[str, Any]:
    """
    Get cached temperature for a city from the database.

    Args:
        city: Name of the city (e.g., "Bratislava", "Košice")
        weather_date: Date to check for (YYYY-MM-DD format). Defaults to today.

    Returns:
        dict with cached data if found and not expired, or indication that cache miss occurred
    """
    try:
        pool = await get_pool()
        # Default to today's date if not provided
        if weather_date is None:
            weather_date = datetime.now().strftime("%Y-%m-%d")

        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute("""
                    SELECT city, latitude, longitude, temperature, weather,
                           weather_date, fetched_at, expires_at,
                           expires_at > NOW() as is_valid
                    FROM temperature_cache
                    WHERE LOWER(city) = LOWER(%s) AND weather_date = %s
                """, (city, weather_date))
                row = await cur.fetchone()

                if row and row['is_valid']:
                    return {
                        "found": True,
                        "cached": True,
                        "city": row['city'],
                        "latitude": float(row['latitude']),
                        "longitude": float(row['longitude']),
                        "temperature": float(row['temperature']),
                        "weather": row['weather'],
                        "weather_date": row['weather_date'].isoformat() if row['weather_date'] else None,
                        "fetched_at": row['fetched_at'].isoformat() if row['fetched_at'] else None,
                        "expires_at": row['expires_at'].isoformat() if row['expires_at'] else None,
                    }
                elif row:
                    return {
                        "found": False,
                        "reason": "cache_expired",
                        "message": f"Cache for {city} on {weather_date} has expired (was valid until {row['expires_at'].isoformat()})",
                        "last_temperature": float(row['temperature']),
                    }
                else:
                    return {
                        "found": False,
                        "reason": "not_in_cache",
                        "message": f"No cached data for {city} on {weather_date}",
                    }
    except Exception as e:
        return {
            "found": False,
            "reason": "error",
            "message": f"Database error: {str(e)}",
        }


@mcp.tool()
async def store_temperature(
    city: str,
    latitude: float,
    longitude: float,
    temperature: float,
    weather: str = None,
    weather_date: str = None,
    cache_hours: int = 1
) -> dict[str, Any]:
    """
    Store temperature data in the database cache.

    Args:
        city: Name of the city
        latitude: Latitude of the location
        longitude: Longitude of the location
        temperature: Current temperature in Celsius
        weather: Weather description (e.g., "Sunny", "Cloudy")
        weather_date: Date of the weather (YYYY-MM-DD format). Defaults to today.
        cache_hours: How long to cache the data (default: 1 hour)

    Returns:
        dict indicating success or failure
    """
    try:
        pool = await get_pool()
        expires_at = datetime.now() + timedelta(hours=cache_hours)
        # Default to today's date if not provided
        if weather_date is None:
            weather_date = datetime.now().strftime("%Y-%m-%d")

        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    INSERT INTO temperature_cache
                        (city, latitude, longitude, temperature, weather, weather_date, fetched_at, expires_at)
                    VALUES (%s, %s, %s, %s, %s, %s, NOW(), %s)
                    ON DUPLICATE KEY UPDATE
                        latitude = VALUES(latitude),
                        longitude = VALUES(longitude),
                        temperature = VALUES(temperature),
                        weather = VALUES(weather),
                        fetched_at = NOW(),
                        expires_at = VALUES(expires_at)
                """, (
                    city,
                    latitude,
                    longitude,
                    temperature,
                    weather,
                    weather_date,
                    expires_at
                ))

                return {
                    "success": True,
                    "message": f"Temperature for {city} on {weather_date} cached successfully",
                    "city": city,
                    "temperature": temperature,
                    "weather": weather,
                    "weather_date": weather_date,
                    "expires_at": expires_at.isoformat(),
                }
    except Exception as e:
        return {
            "success": False,
            "message": f"Failed to store temperature: {str(e)}",
        }


@mcp.tool()
async def get_all_cached_temperatures() -> dict[str, Any]:
    """
    Get all cached temperatures from the database.

    Returns:
        dict with list of all cached temperature data
    """
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute("""
                    SELECT city, latitude, longitude, temperature, weather,
                           weather_date, fetched_at, expires_at,
                           expires_at > NOW() as is_valid
                    FROM temperature_cache
                    ORDER BY city, weather_date
                """)
                rows = await cur.fetchall()

                results = []
                for row in rows:
                    results.append({
                        "city": row['city'],
                        "latitude": float(row['latitude']),
                        "longitude": float(row['longitude']),
                        "temperature": float(row['temperature']),
                        "weather": row['weather'],
                        "weather_date": row['weather_date'].isoformat() if row['weather_date'] else None,
                        "fetched_at": row['fetched_at'].isoformat() if row['fetched_at'] else None,
                        "expires_at": row['expires_at'].isoformat() if row['expires_at'] else None,
                        "is_valid": bool(row['is_valid']),
                    })

                return {
                    "success": True,
                    "count": len(results),
                    "temperatures": results,
                }
    except Exception as e:
        return {
            "success": False,
            "message": f"Database error: {str(e)}",
            "temperatures": [],
        }


@mcp.tool()
async def delete_cached_temperature(city: str) -> dict[str, Any]:
    """
    Delete cached temperature for a specific city.

    Args:
        city: Name of the city to delete from cache

    Returns:
        dict indicating success or failure
    """
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    DELETE FROM temperature_cache
                    WHERE LOWER(city) = LOWER(%s)
                """, (city,))
                deleted_count = cur.rowcount

                return {
                    "success": True,
                    "message": f"Deleted {deleted_count} record(s) for {city}",
                    "deleted_count": deleted_count,
                }
    except Exception as e:
        return {
            "success": False,
            "message": f"Failed to delete: {str(e)}",
        }


@mcp.tool()
async def clear_expired_cache() -> dict[str, Any]:
    """
    Clear all expired entries from the cache.

    Returns:
        dict indicating how many entries were cleared
    """
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    DELETE FROM temperature_cache
                    WHERE expires_at < NOW()
                """)
                deleted_count = cur.rowcount

                return {
                    "success": True,
                    "message": f"Cleared {deleted_count} expired cache entries",
                    "deleted_count": deleted_count,
                }
    except Exception as e:
        return {
            "success": False,
            "message": f"Failed to clear cache: {str(e)}",
        }


if __name__ == "__main__":
    mcp.run(transport='stdio')

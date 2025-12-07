
# Multi-city workflow - LLM determines coordinates for each city
MULTI_CITY_TEMPERATURE_WORKFLOW = {
    "name": "Multi-City Temperature Workflow",
    "description": "Get temperatures for multiple cities with caching - LLM determines coordinates",
    "nodes": [
        {
            "id": "start",
            "type": "start",
            "name": "Start",
            "description": "Begin multi-city temperature lookup",
            "next": ["check_all_caches"],
        },
        {
            "id": "check_all_caches",
            "type": "task",
            "name": "Check All Caches",
            "description": "Use get_all_cached_temperatures to see what cities are already cached with valid data. Then identify which cities from the requested list still need fresh data.",
            "data": {
                "output_var": "cache_status",
            },
            "next": ["fetch_missing_decision"],
        },
        {
            "id": "fetch_missing_decision",
            "type": "condition",
            "name": "Need to Fetch Missing?",
            "description": "Check if there are any cities that need fresh data",
            "condition": "Are there any cities in the requested list that don't have valid cache entries? If yes, condition is true.",
            "next": ["fetch_missing_cities", "compile_results"],  # [true, false]
        },
        {
            "id": "fetch_missing_cities",
            "type": "task",
            "name": "Fetch Missing Cities",
            "description": "For each city that needs fresh data: 1) Determine its latitude/longitude using your geographic knowledge, 2) Call get_current_weather with those coordinates, 3) Call store_temperature to cache the result with the full weather_data. Process all missing cities.",
            "data": {
                "output_var": "fetched_data",
            },
            "next": ["compile_results"],
        },
        {
            "id": "compile_results",
            "type": "task",
            "name": "Compile All Results",
            "description": "Combine all cached data and freshly fetched data into a comprehensive temperature report. For each city, show the temperature and indicate whether it came from cache or was freshly fetched.",
            "data": {
                "output_var": "final_report",
            },
            "next": ["end"],
        },
        {
            "id": "end",
            "type": "end",
            "name": "End",
            "description": "Workflow complete",
            "next": [],
        },
    ],
}


# Example cities (for reference, but LLM should determine coordinates)
EXAMPLE_CITIES = [
    "Bratislava",
    "Košice",
    "Trenčín",
]

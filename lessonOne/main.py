import asyncio
import json
from typing import List, Dict, Any

import python_weather
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(
    base_url = "http://localhost:11434/v1/",  # Ollama’s local endpoint
    api_key = "ollama",  # required by the library but ignored by Ollama locally
)

# Function Implementations
async def get_weather_async(city: str):
    """Get the weather."""
    async with python_weather.Client(unit=python_weather.METRIC) as client:
        # get the weather for a location
        weather = await client.get(city)
        temperature = weather.temperature
        return {"city": city, "Temperature": temperature}

def get_weather(city: str):
    return asyncio.run(get_weather_async(city))

# Define custom tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Use this function to get the temperature for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city in world eg. Bratislava",
                    }
                },
                "required": ["city"],
            },
        },
    }
]

available_functions = {
    "get_weather": get_weather,
}


class Agent:

    def __init__(self, model: str = "qwen3:4b"):
        self.model = model

    def run(self, messages: List[Dict[str, Any]]) -> object:
        # Call the LLM
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            parallel_tool_calls=False,
        )

        print("LLM Response:", response)

        if response.choices[0].message.tool_calls:

            # Find the tool call content
            tool_call = response.choices[0].message.tool_calls[0]

            # Extract tool name and arguments
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            # Call the function
            function_to_call = available_functions[function_name]
            function_response = function_to_call(**function_args)

            print("\nFunction response:", function_response)

            messages.append({
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": function_name,
                            "arguments": json.dumps(function_args),
                        }
                    }
                ]
            })
            messages.append({
                "role": "tool",
                "name": function_name,
                "content": json.dumps(function_response),
            })

            # Second call to get final response based on function output
            second_response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )
            final_answer = second_response.choices[0].message

            print("Second response:", final_answer)
            return final_answer
        else:
            print("No tool calls returned from model")

        return "Something is wrong sorry no sorry"


def main():
    # Create a ReAct agent
    agent = Agent()

    # Example 1: Simple query (single tool call)
    print("=== Example 1: Single Tool Call ===")
    messages1 = [
        {"role": "system", "content": "You are a helpful AI assistant. "},
        {"role": "user", "content": "What is the current weather in Bratislava?"},
    ]

    result1 = agent.run(messages1.copy())
    print(f"\nResult: {result1.content}")

if __name__ == "__main__":
    main()
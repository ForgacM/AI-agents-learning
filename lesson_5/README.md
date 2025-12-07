# Lesson five AI tools
- Lite LLM to unified access to any LLM model providers
  - only one library to rule them all
- React agent (Reason - act) 
  - great for reasoning, in custom framework everything must be implemented manually
  - You have all logs, all implementation details in hand

## How to run

```
cd lesson_5

uv venv
uv venv --python 3.12  
uv sync

uv run -m src.react_agent
uv run -m src.temperature_workflow

```

### Prerequisites 
MCP server running -> from lesson_2 server_https.py
In case of using LiteLLM than check README.md in lite_llm folder


## Lessons Learned
- Lite LLM tool in one hand helping with multimodel support but on the other hand you have problem 
with debugging and finding why your model is not answering as via direct API call 
- Workflow in custom workflow framework is excellent if I want to generate workflows using LLM or claude code,
but on the other hand there is lots of code to handle all edge cases as well as to support multiple mcp etc.
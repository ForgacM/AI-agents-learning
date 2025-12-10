# Lesson six AI_agents
- using langchain and langgraph frameworks
- manage flows in langgraph
- write agents using langchain

## How to run

```
 lesson_6 % uv run -m 02_langgraph.main
```

## Prerequisites

```
 lesson_5 % uv run -m mcp_servers.run_db_server
 lesson_2 % uv run server_https.py
 lesson_6 % uv python install 3.12     
 uv venv --python 3.12
 uv sync
```

## Lessons Learned
- Langchain is powerful and easy to use for basic agents and also
react agents. Fast and direct approach to use tools or mcp servers
no need to create a lot of service classes. Working nice out of the box
- Langgraph its crucial to write good prompt for LLM to work as you want 
this is little bit tricky compare from custom workflows.
- Langgraph nice and straightforward graph builder and compiler. 
- Each agent must be expert in one domain and in system promt you have to specify which agents he can only use
- 
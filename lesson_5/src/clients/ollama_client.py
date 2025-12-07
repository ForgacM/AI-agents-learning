"""
Direct Ollama client - bypasses LiteLLM for better Qwen3 thinking/tool support.

This client provides OpenAI-compatible response format while calling Ollama directly.
"""

import json
import logging
import os
from typing import List, Dict, Any, Optional, TypeVar

import httpx
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall, Function
from openai.types.completion_usage import CompletionUsage
from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)


class OllamaClient:
    """
    Direct Ollama client with OpenAI-compatible response format.

    Features:
    - Proper handling of Qwen3's thinking/reasoning field
    - Correct tool call argument conversion (dict → JSON string)
    - Drop-in replacement for LLMClient in ReAct agent

    Usage:
        client = OllamaClient(model="qwen3:4b")
        response = await client.call(messages, tools=tools)
    """

    def __init__(
        self,
        model: str = "qwen3:4b",
        base_url: Optional[str] = None,
        timeout: float = 1200.0,
    ):
        """
        Initialize Ollama client.

        Args:
            model: Ollama model name (without 'ollama/' prefix)
            base_url: Ollama server URL (default: from OLLAMA_BASE_URL env var)
            timeout: Request timeout in seconds
        """
        self.model = model.replace("ollama/", "") if model.startswith("ollama/") else model
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.client = httpx.AsyncClient(timeout=timeout)

        logger.info(f"✅ OllamaClient initialized: model={self.model}, base_url={self.base_url}")

    async def call(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> ChatCompletion:
        """
        Generate chat completion - OpenAI compatible response.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            tools: OpenAI-format tool definitions

        Returns:
            OpenAI ChatCompletion object
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }

        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        if tools:
            payload["tools"] = tools

        try:
            logger.info(f"Calling Ollama: {self.base_url}/api/chat with model={self.model}")

            response = await self.client.post(
                f"{self.base_url}/api/chat",
                json=payload,
            )
            response.raise_for_status()
            result = response.json()

            logger.debug(f"Ollama raw response: {json.dumps(result, indent=2)}")

            return self._to_openai_format(result)

        except httpx.ConnectError as e:
            error_msg = f"Cannot connect to Ollama at {self.base_url}. Is Ollama running? Error: {e}"
            logger.error(f"❌ {error_msg}")
            raise ConnectionError(error_msg) from e
        except httpx.HTTPStatusError as e:
            error_msg = f"Ollama HTTP error: {e.response.status_code} - {e.response.text}"
            logger.error(f"❌ {error_msg}")
            raise RuntimeError(error_msg) from e
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON response from Ollama: {e}"
            logger.error(f"❌ {error_msg}")
            raise RuntimeError(error_msg) from e
        except Exception as e:
            error_msg = f"Ollama call error: {type(e).__name__}: {e}"
            logger.error(f"❌ {error_msg}")
            import traceback
            logger.error(traceback.format_exc())
            raise RuntimeError(error_msg) from e

    def _to_openai_format(self, ollama_response: Dict[str, Any]) -> ChatCompletion:
        """Convert Ollama response to OpenAI ChatCompletion format."""

        message = ollama_response.get("message", {})

        # Handle content and thinking
        content = message.get("content", "")
        thinking = message.get("thinking", "")

        # Qwen3 puts reasoning in 'thinking' field when using tools
        # If content is empty but thinking exists, use thinking
        if not content and thinking:
            logger.debug("Using 'thinking' field as content (Qwen3 reasoning mode)")
            content = thinking

        # Convert tool calls from Ollama format to OpenAI format
        tool_calls = self._convert_tool_calls(message.get("tool_calls"))

        # Build ChatCompletionMessage
        chat_message = ChatCompletionMessage(
            role="assistant",
            content=content if content else None,
            tool_calls=tool_calls,
        )

        # Build Choice
        choice = Choice(
            index=0,
            message=chat_message,
            finish_reason=ollama_response.get("done_reason", "stop"),
        )

        # Build usage info
        usage = CompletionUsage(
            prompt_tokens=ollama_response.get("prompt_eval_count", 0),
            completion_tokens=ollama_response.get("eval_count", 0),
            total_tokens=(
                ollama_response.get("prompt_eval_count", 0) +
                ollama_response.get("eval_count", 0)
            ),
        )

        # Build ChatCompletion
        return ChatCompletion(
            id=f"chatcmpl-ollama-{id(ollama_response)}",
            model=ollama_response.get("model", self.model),
            object="chat.completion",
            created=0,
            choices=[choice],
            usage=usage,
        )

    def _convert_tool_calls(
        self,
        ollama_tool_calls: Optional[List[Dict[str, Any]]]
    ) -> Optional[List[ChatCompletionMessageToolCall]]:
        """Convert Ollama tool calls to OpenAI format."""

        if not ollama_tool_calls:
            return None

        tool_calls = []
        for i, tc in enumerate(ollama_tool_calls):
            func = tc.get("function", {})

            # Ollama returns arguments as dict, OpenAI expects JSON string
            arguments = func.get("arguments", {})
            if isinstance(arguments, dict):
                arguments = json.dumps(arguments)

            tool_calls.append(
                ChatCompletionMessageToolCall(
                    id=tc.get("id", f"call_{i}"),
                    type="function",
                    function=Function(
                        name=func.get("name", ""),
                        arguments=arguments,
                    )
                )
            )

        return tool_calls if tool_calls else None

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
        logger.info("OllamaClient closed")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

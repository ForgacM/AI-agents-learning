import json
import logging
import os
from typing import List, Dict, Any, Optional, Type, TypeVar

import httpx
import instructor
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall, Function
from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)


class LLMClient:
    """
    LLM client with support for:
    - LiteLLM proxy (default)
    - Direct Ollama API (for better Qwen3 thinking/tool support)
    """

    def __init__(
        self,
        llm_model: str = "ollama/qwen3:4b",
        use_direct_ollama: bool = True,  # Use direct Ollama by default for better compatibility
    ):
        self.llm_model = llm_model
        self.use_direct_ollama = use_direct_ollama and llm_model.startswith("ollama/")

        # LiteLLM settings
        self.litellm_base_url = os.getenv("LITELLM_BASE_URL", "http://0.0.0.0:4000")
        self.litellm_api_key = os.getenv("LITELLM_API_KEY", "dummy-key")

        # Direct Ollama settings
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://192.168.1.200:11434")
        self.ollama_model = llm_model.replace("ollama/", "") if llm_model.startswith("ollama/") else llm_model

        self._initialize_clients()

    def _initialize_clients(self):
        """Initialize the appropriate clients."""
        try:
            # Always initialize LiteLLM client (for non-Ollama models or fallback)
            self.litellm_client = AsyncOpenAI(
                api_key=self.litellm_api_key,
                base_url=self.litellm_base_url,
            )

            # Instructor client for structured outputs
            # Use TOOLS mode for Anthropic compatibility (JSON mode requires tools param)
            self.instructor_client = instructor.from_openai(
                self.litellm_client,
                mode=instructor.Mode.TOOLS
            )

            # HTTP client for direct Ollama calls
            self.http_client = httpx.AsyncClient(timeout=120.0)

            if self.use_direct_ollama:
                logger.info(f"✅ LLM client initialized - using DIRECT Ollama at {self.ollama_base_url}")
            else:
                logger.info(f"✅ LLM client initialized - using LiteLLM at {self.litellm_base_url}")

        except Exception as e:
            logger.error(f"❌ Failed to initialize LLM client: {e}")
            raise

    async def call(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> ChatCompletion:
        """Generate chat response."""
        if self.use_direct_ollama:
            return await self._call_ollama_direct(messages, temperature, max_tokens, tools)
        else:
            return await self._call_litellm(messages, temperature, max_tokens, tools)

    async def _call_ollama_direct(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> ChatCompletion:
        """Call Ollama API directly - better support for Qwen3 thinking/tools."""

        payload = {
            "model": self.ollama_model,
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
            response = await self.http_client.post(
                f"{self.ollama_base_url}/api/chat",
                json=payload,
            )
            response.raise_for_status()
            result = response.json()

            # Convert Ollama response to OpenAI ChatCompletion format
            return self._convert_ollama_to_openai(result)

        except Exception as e:
            logger.error(f"❌ Ollama direct call error: {e}")
            raise

    def _convert_ollama_to_openai(self, ollama_response: Dict[str, Any]) -> ChatCompletion:
        """Convert Ollama response format to OpenAI ChatCompletion format."""

        message = ollama_response.get("message", {})

        # Get content - merge thinking if content is empty
        content = message.get("content", "")
        thinking = message.get("thinking", "")

        # If content is empty but we have thinking, use thinking as content
        # This handles Qwen3's thinking mode
        if not content and thinking:
            content = thinking

        # Convert tool calls if present
        tool_calls = None
        ollama_tool_calls = message.get("tool_calls")

        if ollama_tool_calls:
            tool_calls = []
            for tc in ollama_tool_calls:
                func = tc.get("function", {})
                # Ollama returns arguments as dict, OpenAI expects JSON string
                arguments = func.get("arguments", {})
                if isinstance(arguments, dict):
                    arguments = json.dumps(arguments)

                tool_calls.append(
                    ChatCompletionMessageToolCall(
                        id=tc.get("id", f"call_{len(tool_calls)}"),
                        type="function",
                        function=Function(
                            name=func.get("name", ""),
                            arguments=arguments,
                        )
                    )
                )

        # Build the ChatCompletionMessage
        chat_message = ChatCompletionMessage(
            role="assistant",
            content=content if content else None,
            tool_calls=tool_calls if tool_calls else None,
        )

        # Build the Choice
        choice = Choice(
            index=0,
            message=chat_message,
            finish_reason=ollama_response.get("done_reason", "stop"),
        )

        # Build the ChatCompletion
        return ChatCompletion(
            id=f"chatcmpl-ollama-{ollama_response.get('created_at', '')}",
            model=ollama_response.get("model", self.ollama_model),
            object="chat.completion",
            created=int(ollama_response.get("created_at", "0").split(".")[0].replace("-", "").replace(":", "").replace("T", "")[:10]) if ollama_response.get("created_at") else 0,
            choices=[choice],
        )

    async def _call_litellm(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> ChatCompletion:
        """Call via LiteLLM proxy."""
        try:
            kwargs = {
                "model": self.llm_model,
                "messages": messages,
                "temperature": temperature,
                "stream": False,
            }

            if max_tokens:
                kwargs["max_tokens"] = max_tokens

            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"

            response = await self.litellm_client.chat.completions.create(**kwargs)
            return response

        except Exception as e:
            logger.error(f"❌ LiteLLM chat error: {e}")
            raise

    async def call_structured(
        self,
        messages: List[Dict[str, str]],
        response_model: Type[T],
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        max_retries: int = 3,
    ) -> T:
        """Generate structured response using Instructor."""
        try:
            response = await self.instructor_client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                response_model=response_model,
                temperature=temperature,
                max_tokens=max_tokens,
                max_retries=max_retries,
            )

            return response

        except Exception as e:
            logger.error(f"❌ Instructor structured output error: {e}")
            raise

    async def close(self):
        """Close HTTP client."""
        if hasattr(self, 'http_client'):
            await self.http_client.aclose()

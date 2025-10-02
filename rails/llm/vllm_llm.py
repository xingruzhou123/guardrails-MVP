import httpx
from typing import List, Dict, AsyncIterator
import asyncio


class VLLMChatLLM:
    """
    Adapter for vLLM running via llama-serve (OpenAI-compatible API).
    Implements acomplete and astream like HFChatLLM.
    """

    def __init__(
        self, base_url="http://localhost:8000/v1", model="Qwen/Qwen3-4B-Thinking-2507"
    ):
        self.base_url = base_url
        self.model = model

    async def acomplete(self, messages: List[Dict[str, str]], **kwargs) -> str:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": kwargs.get("temperature", 0.7),
                    "max_tokens": kwargs.get("max_new_tokens", 512),
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

    async def astream(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> AsyncIterator[str]:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": kwargs.get("temperature", 0.7),
                    "max_tokens": kwargs.get("max_new_tokens", 512),
                    "stream": True,
                },
            ) as resp:
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        chunk = line[len("data: ") :].strip()
                        if chunk == "[DONE]":
                            break
                        try:
                            data = httpx.Response(200, content=chunk).json()
                            delta = data["choices"][0]["delta"].get("content")
                            if delta:
                                yield delta
                        except Exception:
                            continue

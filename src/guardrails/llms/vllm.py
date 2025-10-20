import asyncio
from openai import OpenAI
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid
from typing import List, Dict, AsyncIterator

from .base import BaseLLM


class VLLMOpenAIClient(BaseLLM):
    def __init__(self, base_url: str, model_name: str):
        self.client = OpenAI(base_url=base_url, api_key="EMPTY")
        self.model = model_name

    async def acomplete(self, messages: List[Dict[str, str]], **kwargs) -> str:
        # ... (full content of acomplete)
        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=kwargs.get("max_tokens", 512),
                    temperature=kwargs.get("temperature", 0.1),
                    stream=False,
                ),
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            print(
                f"ERROR: Could not connect to vLLM server at {self.client.base_url}. Details: {e}"
            )
            return "Error: The generation model is currently unavailable."

    async def astream(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> AsyncIterator[str]:
        full_response = await self.acomplete(messages, **kwargs)
        yield full_response


class VLLMInProcessLLM(BaseLLM):
    def __init__(self, model_name: str):
        engine_args = AsyncEngineArgs(
            model=model_name,
            trust_remote_code=True,
            enforce_eager=True,
            tensor_parallel_size=1,
            distributed_executor_backend="mp",
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.model_name = model_name

    async def acomplete(self, messages: List[Dict[str, str]], **kwargs) -> str:
        # ... (full content of acomplete)
        sampling_params = SamplingParams(
            max_tokens=kwargs.get("max_tokens", 512),
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 0.95),
        )
        tokenizer = await self.engine.get_tokenizer()
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        request_id = f"cmpl-{random_uuid()}"
        results_generator = self.engine.generate(prompt, sampling_params, request_id)
        final_output = ""
        async for request_output in results_generator:
            final_output = request_output.outputs[0].text
        if prompt in final_output:
            return final_output[len(prompt) :]
        return final_output

    async def astream(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> AsyncIterator[str]:
        # ... (full content of astream)
        sampling_params = SamplingParams(
            max_tokens=kwargs.get("max_tokens", 512),
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 0.95),
        )
        tokenizer = await self.engine.get_tokenizer()
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        request_id = f"cmpl-{random_uuid()}"
        results_generator = self.engine.generate(prompt, sampling_params, request_id)
        last_text = ""
        async for request_output in results_generator:
            current_text = request_output.outputs[0].text
            delta = current_text[len(last_text) :]
            last_text = current_text
            if delta:
                yield delta

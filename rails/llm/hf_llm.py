# hf_llm.py
#this code is used to provide a Hugging Face LLM adapter with optional streaming via TextIteratorStreamer.
from __future__ import annotations
from threading import Thread
from typing import Any, AsyncIterator, Dict, List, Optional
from .utils import to_chat_text, run_blocking

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    GenerationConfig,
)


class HFChatLLM:
    """
    Hugging Face LLM adapter with optional streaming via TextIteratorStreamer.
    Implements the BaseLLM interface expected by your rails:
      - acomplete(messages) -> str
      - astream(messages) -> async iterator[str]
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-4B-Thinking-2507",
        device_map: str | dict = "auto",
        torch_dtype: Optional[torch.dtype] = None,  # e.g., torch.bfloat16
        trust_remote_code: bool = True,
        attn_implementation: Optional[str] = None,  # "flash_attention_2" if supported
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.05,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation,
        )
        self.gcfg = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=True if temperature and temperature > 0 else False,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )

    async def acomplete(self, messages: List[Dict[str, str]], **kwargs) -> str:
        prompt = to_chat_text(self.tokenizer, messages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        def _generate_blocking():
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    generation_config=self.gcfg,
                )
            return self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        return await run_blocking(_generate_blocking)

    async def astream(self, messages: List[Dict[str, str]], **kwargs) -> AsyncIterator[str]:
        """
        Token streaming using TextIteratorStreamer in a background thread.
        """
        prompt = to_chat_text(self.tokenizer, messages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_special_tokens=True, skip_prompt=True
        )

        def _worker():
            with torch.no_grad():
                self.model.generate(
                    **inputs,
                    generation_config=self.gcfg,
                    streamer=streamer,
                )

        thread = Thread(target=_worker, daemon=True)
        thread.start()

        try:
            for token_text in streamer:
                yield token_text
        finally:
            thread.join(timeout=5)  # wait for worker to finish or timeout

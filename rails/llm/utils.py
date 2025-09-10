# utils.py
# this code is uesd to provide some utility functions for llmrails
from __future__ import annotations
import asyncio
import functools
import time
from typing import Any, Dict, List, Optional, Tuple

def timeit_async(fn):
    """Measure async function latency; returns (result, seconds)."""
    @functools.wraps(fn)
    async def _wrap(*args, **kwargs):
        t0 = time.time()
        out = await fn(*args, **kwargs)
        return out, time.time() - t0
    return _wrap

def strip_none(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}

def to_chat_text(tokenizer, messages: List[Dict[str, str]]) -> str:
    """
    Use HF chat template if available; otherwise make a simple prompt.
    """
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,  # tells model we're ready for assistant turn
        )
    # Fallback: naive concatenation
    parts = []
    for m in messages:
        role = m.get("role", "user")
        parts.append(f"[{role.upper()}] {m['content']}")
    parts.append("[ASSISTANT] ")
    return "\n".join(parts)

def truncate_to_max_tokens(tokenizer, text: str, max_input_tokens: int) -> str:
    if max_input_tokens is None:
        return text
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_input_tokens:
        return text
    # decode only the last max_input_tokens to preserve the end of prompt if needed
    kept = ids[-max_input_tokens:]
    return tokenizer.decode(kept, skip_special_tokens=True)

def approx_token_count(tokenizer, text: str) -> int:
    try:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])
    except Exception:
        # crude fallback ~ 4 chars per token
        return max(1, len(text) // 4)

async def run_blocking(func, *args, loop=None, **kwargs):
    """
    Run a blocking function in a thread from async code.
    """
    loop = loop or asyncio.get_event_loop()
    return await loop.run_in_executor(None, functools.partial(func, *args, **kwargs))

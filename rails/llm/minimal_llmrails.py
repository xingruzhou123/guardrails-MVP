# minimal_llmrails.py
# this code is used to provide a minimal implementation of llmrails with echo llm and hf llm adapter
from __future__ import annotations
import asyncio
import functools
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Tuple

# ========= Config =========

@dataclass
class ModelConfig:
    provider: str = "echo"
    name: str = "amd/Instella-3B"
    streaming: bool = True
    params: Dict[str, Any] = field(default_factory=dict)
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.05
    device_map: Any = "auto"
    dtype: Optional[str] = None        # "bfloat16", "float16", or None -> auto
    trust_remote_code: bool = True
    attn_implementation: Optional[str] = None


@dataclass
class RailsConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    streaming_enabled: bool = True
    tracing_enabled: bool = False
    # regieterred output filters by name (see below)
    output_filters: List[str] = field(default_factory=list)
    # whether to enable tool calls
    tools_enabled: bool = False

# ========= Minimal LLM Interface & Dummy Implementation =========

class BaseLLM:
    async def acomplete(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Return a full completion string."""
        raise NotImplementedError

    async def astream(self, messages: List[Dict[str, str]], **kwargs) -> AsyncIterator[str]:
        """Yield tokens/chunks one by one."""
        # default impl: split by whitespace
        text = await self.acomplete(messages, **kwargs)
        for i, ch in enumerate(text.split()):
            yield ("" if i == 0 else " ") + ch

class EchoLLM(BaseLLM):
    """A dummy LLM that just echoes the last user message."""
    async def acomplete(self, messages: List[Dict[str, str]], **kwargs) -> str:
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        await asyncio.sleep(0)  # Yield control
        return f"[echo] {last_user}"

# ========= Small utilities (no external files) =========

def to_chat_text(tokenizer, messages: List[Dict[str, str]]) -> str:
    """
    Prefer the tokenizer's chat template if available; otherwise fallback.
    """
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            # No chat_template set (or other template failure) -> fallback
            pass

    # Fallback: simple role-tagged concatenation
    parts = []
    for m in messages:
        role = m.get("role", "user").upper()
        parts.append(f"[{role}] {m['content']}")
    parts.append("[ASSISTANT] ")
    return "\n".join(parts)


async def run_blocking(func, *args, loop=None, **kwargs):
    """
    Run a blocking function in a thread from async code.
    """
    loop = loop or asyncio.get_event_loop()
    return await loop.run_in_executor(None, functools.partial(func, *args, **kwargs))

# ========= Hugging Face Adapter (Instella-ready) =========

class HFChatLLM(BaseLLM):
    """
    Hugging Face LLM adapter with optional streaming via TextIteratorStreamer.
    Only imported if 'transformers' and 'torch' are available.
    """

    def __init__(
        self,
        model_name: str = "amd/Instella-3B",
        device_map: Any = "auto",
        dtype: Optional[str] = None,           # "bfloat16" | "float16" | None
        trust_remote_code: bool = True,
        attn_implementation: Optional[str] = None,   # e.g., "flash_attention_2"
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.05,
    ):
        try:
            import torch  # type: ignore
            from transformers import (  # type: ignore
                AutoModelForCausalLM,
                AutoTokenizer,
                TextIteratorStreamer,
                GenerationConfig,
            )
        except Exception as e:
            raise RuntimeError(
                "HFChatLLM requires 'torch' and 'transformers' to be installed."
            ) from e

        # dtype parsing
        torch_dtype = None
        if dtype:
            d = dtype.lower()
            if d == "bfloat16":
                torch_dtype = torch.bfloat16
            elif d in ("float16", "fp16", "half"):
                torch_dtype = torch.float16
        self._torch = torch
        self._TextIteratorStreamer = TextIteratorStreamer
        self._AutoTokenizer = AutoTokenizer
        self._AutoModelForCausalLM = AutoModelForCausalLM
        self._GenerationConfig = GenerationConfig

        self.tokenizer = self._AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        self.model = self._AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            # transformers still expects the kwarg name 'torch_dtype'
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation,
        )
        self.gcfg = self._GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=True if temperature and temperature > 0 else False,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        

    async def acomplete(self, messages: List[Dict[str, str]], **kwargs) -> str:
        torch = self._torch
        prompt = to_chat_text(self.tokenizer, messages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        def _generate_blocking():
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    generation_config=self.gcfg,
                )
            # slice out only the newly generated tokens
            return self.tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )

        return await run_blocking(_generate_blocking)

    async def astream(self, messages: List[Dict[str, str]], **kwargs) -> AsyncIterator[str]:
        """
        Token streaming using TextIteratorStreamer in a background thread.
        """
        from threading import Thread

        torch = self._torch
        TextIteratorStreamer = self._TextIteratorStreamer

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

        for token_text in streamer:
            # token_text is sync-iterable; wrap as async
            yield token_text

# ========= Filters (Guardrails) =========

class BaseFilter:
    def filter(self, text: str, context: Dict[str, Any]) -> Tuple[bool, Optional[str], str]:
        """
        返回: (allowed, reason, possibly_modified_text)
        """
        raise NotImplementedError

class MaxLengthFilter(BaseFilter):
    def __init__(self, max_chars: int = 2000):
        self.max_chars = max_chars

    def filter(self, text: str, context: Dict[str, Any]) -> Tuple[bool, Optional[str], str]:
        if len(text) > self.max_chars:
            return False, f"output too long: {len(text)}>{self.max_chars}", text[: self.max_chars]
        return True, None, text

class BannedWordsFilter(BaseFilter):
    def __init__(self, banned: List[str]):
        self.banned = banned

    def filter(self, text: str, context: Dict[str, Any]) -> Tuple[bool, Optional[str], str]:
        lowered = text.lower()
        for w in self.banned:
            if w.lower() in lowered:
                return False, f"banned word: {w}", text
        return True, None, text

# ========= Action / Tool Registry (Optional) =========

ToolFn = Callable[[Dict[str, Any]], Any]

class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, ToolFn] = {}

    def register(self, name: str, fn: ToolFn):
        self._tools[name] = fn

    def get(self, name: str) -> Optional[ToolFn]:
        return self._tools.get(name)

# ========= Minimal Runtime (no Colang) =========

class SimpleRuntime:
    """
    extreme simple pipeline：
    1) pre-processing(optional,not yet)
    2) call LLM
    3) Analysis tool/action (optional) / call LLM (optional)
    4) Output filtering
    """
    def __init__(self, config: RailsConfig, llm: BaseLLM,
                 filters: Dict[str, BaseFilter],
                 tools: ToolRegistry):
        self.config = config
        self.llm = llm
        self.filters = filters
        self.tools = tools

    async def run_once(self, messages: List[Dict[str, str]], context: Dict[str, Any]) -> str:
        # 1) pre-processing(optional,not yet)
        # 2) call LLM
        output = await self.llm.acomplete(messages)

        # 3) Analysis tool/action (optional example: if the output contains the {{tool:name json}} structure)
        if self.config.tools_enabled and "{{tool:" in output:
            try:
                name, payload = self._parse_tool_call(output)
                fn = self.tools.get(name)
                if fn:
                    result = fn({"messages": messages, "context": context, "payload": payload})
                    output = f"{output}\n\n(tool result) {result}"
            except Exception as e:
                output = f"{output}\n\n(tool error) {e}"

        # 4) Output filtering (sequential execution)
        allowed, reason, out = self._apply_filters(output, context)
        if not allowed:
            # can choose to truncate/replace/report an error. Here are the reasons and the text that was passively truncated.
            return f"[blocked] {reason}\n{out}"
        return out

    async def run_stream(self, messages: List[Dict[str, str]], context: Dict[str, Any]) -> AsyncIterator[str]:
        """
        When streaming output, it is also necessary to apply filters to each chunk. 
        To avoid misjudging half a word, create a simple buffer based on whitespace for tokenization.
        """
        buffer_text = ""
        async for chunk in self.llm.astream(messages):
            buffer_text += chunk
            # only flush when we end on whitespace or punctuation likely to end a token
            flush_now = len(buffer_text) >= 1 and buffer_text[-1:].isspace()
            if not flush_now:
                # small heuristic: also flush if buffer is large to avoid latency
                flush_now = len(buffer_text) > 64

            if flush_now:
                allowed, reason, mod = self._apply_filters(buffer_text, context)
                if not allowed:
                    yield f"[blocked] {reason}\n{mod}"
                    return
                yield buffer_text
                buffer_text = ""

        # tail
        if buffer_text:
            allowed, reason, mod = self._apply_filters(buffer_text, context)
            if not allowed:
                yield f"[blocked] {reason}\n{mod}"
                return
            yield buffer_text

    def _apply_filters(self, text: str, context: Dict[str, Any]) -> Tuple[bool, Optional[str], str]:
        current = text
        for name in self.config.output_filters:
            f = self.filters.get(name)
            if not f:
                continue
            ok, reason, current = f.filter(current, context)
            if not ok:
                return False, reason, current
        return True, None, current

    def _parse_tool_call(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """
            Accept 3 formats:
            1) {{tool:NAME {...}}}
            2) {TOOL:NAME {...}}
            3) ```json {"tool":"NAME","args":{...}} ```
            Returns (name, payload_dict)
        """
        import json, re
        m = re.search(r"\{\{tool:([a-zA-Z0-9_]+)\s+(\{.*?\})\}\}", text, re.DOTALL)
        if m:
            return m.group(1), json.loads(m.group(2))
        m = re.search(r"\{TOOL:([a-zA-Z0-9_]+)\s*(\{.*?\})\}", text, re.DOTALL)
        if m:
            return m.group(1), json.loads(m.group(2))
        m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
        if m:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict) and "tool" in obj:
                name = obj["tool"]
                payload = obj.get("args", {}) or obj.get("payload", {}) or {}
                if not isinstance(payload, dict):
                    payload = {}
                return name, payload
        raise ValueError("no valid tool call found")

# ========= Public Rails API =========

class LLMRails:
    def __init__(self, config: RailsConfig, llm: Optional[BaseLLM] = None):
        self.config = config
        self.llm = llm or self._build_llm_from_config(config.model)
        self.filters: Dict[str, BaseFilter] = {}
        self.tools = ToolRegistry()
        # default filters
        self.register_filter("max_length", MaxLengthFilter(max_chars=4000))
        self.register_filter("banned_words", BannedWordsFilter(banned=[]))
        self.runtime = SimpleRuntime(config=self.config, llm=self.llm,
                                     filters=self.filters, tools=self.tools)

    # ---- Registration APIs ----
    def register_filter(self, name: str, filter_obj: BaseFilter):
        self.filters[name] = filter_obj

    def register_tool(self, name: str, fn: ToolFn):
        self.tools.register(name, fn)

    def update_llm(self, llm: BaseLLM):
        self.llm = llm
        self.runtime.llm = llm

    # ---- Generate APIs ----
    async def generate_async(self, messages: List[Dict[str, str]], context: Optional[Dict[str, Any]] = None) -> str:
        context = context or {}
        t0 = time.time()
        out = await self.runtime.run_once(messages, context)
        if self.config.tracing_enabled:
            print(f"[trace] took={time.time()-t0:.3f}s")
        return out

    def generate(self, messages: List[Dict[str, str]], context: Optional[Dict[str, Any]] = None) -> str:
        return asyncio.run(self.generate_async(messages, context))

    async def stream_async(self, messages: List[Dict[str, str]], context: Optional[Dict[str, Any]] = None) -> AsyncIterator[str]:
        context = context or {}
        async for ch in self.runtime.run_stream(messages, context):
            yield ch

    # ---- Private helpers ----
    def _build_llm_from_config(self, mc: ModelConfig) -> BaseLLM:
        if mc.provider.lower() == "hf":
            return HFChatLLM(
                model_name=mc.name,
                device_map=mc.device_map,
                dtype=mc.dtype,
                trust_remote_code=mc.trust_remote_code,
                attn_implementation=mc.attn_implementation,
                max_new_tokens=mc.max_new_tokens,
                temperature=mc.temperature,
                top_p=mc.top_p,
                repetition_penalty=mc.repetition_penalty,
            )
        return EchoLLM()


# ========= (optional) quick demo =========
if __name__ == "__main__":
    # Example: switch to Hugging Face Instella
    cfg = RailsConfig(
    model=ModelConfig(
        provider="hf",
        name="amd/Instella-3B-Instruct",  # <-- was "amd/Instella-3B"
        dtype="bfloat16",                 # see patch 3 about dtype
        device_map="auto",
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
    ),
    streaming_enabled=True,
    tracing_enabled=True,
    output_filters=["max_length", "banned_words"],
    tools_enabled=True,
)

    rails = LLMRails(cfg)

    # register a toy tool
    rails.register_tool("sum", lambda ctx: sum(ctx["payload"]["nums"]))

    prompt = [{"role": "user", "content": "say hi to me and call this tool {{tool:sum {\"nums\":[1,2,3,4]}}}"}]

    # non-stream example
    print("== generate ==")
    print(rails.generate(prompt))

    # stream example
    async def demo_stream():
        print("\n== stream ==")
        async for t in rails.stream_async([{"role":"user","content":"using a sentense to introduce AMD"}]):
            print(t, end="", flush=True)
        print()

    asyncio.run(demo_stream())

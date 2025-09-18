# rails/llm/minimal_llmrails.py
# Modified to be a flexible, configuration-driven guardrails system.

from __future__ import annotations
import asyncio
import functools
import time
import re
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Tuple

# New imports for dynamic loading and refactored rules
import yaml
from .rules import BaseOutputRule, RegexRule, SemanticSafetyHTTPRule, OutputRuleResult

# --- AMDLeakGuard and other hardcoded rules have been REMOVED ---
# They are now replaced by the generic RegexRule in rules.py, loaded from config.

# ========= Config (dataclasses must come early; define only once) ============
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
    dtype: Optional[str] = None
    trust_remote_code: bool = True
    attn_implementation: Optional[str] = None

@dataclass
class RailsConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    streaming_enabled: bool = True
    tracing_enabled: bool = False
    # These are now loaded dynamically, but can be kept for other filter types
    output_filters: List[str] = field(default_factory=list)
    system_prompt: Optional[str] = "You are a helpful assistant. Reply in English only."

# ========= Minimal LLM Interface & Dummy Implementation ======================
class BaseLLM:
    async def acomplete(self, messages: List[Dict[str, str]], **kwargs) -> str:
        raise NotImplementedError

    async def astream(self, messages: List[Dict[str, str]], **kwargs) -> AsyncIterator[str]:
        text = await self.acomplete(messages, **kwargs)
        for i, ch in enumerate(text.split()):
            yield ("" if i == 0 else " ") + ch
        await asyncio.sleep(0) # Yield control

class EchoLLM(BaseLLM):
    async def acomplete(self, messages: List[Dict[str, str]], **kwargs) -> str:
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        await asyncio.sleep(0)
        return f"[echo] {last_user}"

# ========= Small utilities ===================================================
def to_chat_text(tokenizer, messages: List[Dict[str, str]]) -> str:
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    parts = []
    for m in messages:
        role = m.get("role", "user").upper()
        parts.append(f"[{role}] {m['content']}")
    parts.append("[ASSISTANT] ")
    return "\n".join(parts)

async def run_blocking(func, *args, loop=None, **kwargs):
    loop = loop or asyncio.get_event_loop()
    return await loop.run_in_executor(None, functools.partial(func, *args, **kwargs))

# ========= Hugging Face Adapter (Instella-ready) =============================
class HFChatLLM(BaseLLM):
    def __init__(
        self,
        model_name: str = "amd/Instella-3B",
        device_map: Any = "auto",
        dtype: Optional[str] = None,
        trust_remote_code: bool = True,
        attn_implementation: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.05,
    ):
        try:
            import torch
            from transformers import (
                AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, GenerationConfig
            )
        except Exception as e:
            raise RuntimeError("HFChatLLM requires 'torch' and 'transformers' to be installed.") from e

        torch_dtype = None
        if dtype:
            d = dtype.lower()
            if d == "bfloat16":
                torch_dtype = torch.bfloat16
            elif d in ("float16", "fp16", "half"):
                torch_dtype = torch.float16

        self._torch = torch
        self._TextIteratorStreamer = TextIteratorStreamer
        self._GenerationConfig = GenerationConfig

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
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
        prompt = to_chat_text(self.tokenizer, messages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        def _generate_blocking():
            with self._torch.no_grad():
                out = self.model.generate(**inputs, generation_config=self.gcfg)
            return self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        return await run_blocking(_generate_blocking)

    async def astream(self, messages: List[Dict[str, str]], **kwargs) -> AsyncIterator[str]:
        from threading import Thread

        prompt = to_chat_text(self.tokenizer, messages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        streamer = self._TextIteratorStreamer(self.tokenizer, skip_special_tokens=True, skip_prompt=True)

        def _worker():
            with self._torch.no_grad():
                self.model.generate(**inputs, generation_config=self.gcfg, streamer=streamer)

        thread = Thread(target=_worker, daemon=True)
        thread.start()

        for token_text in streamer:
            yield token_text

# ========= Minimal Runtime ====================================================
class SimpleRuntime:
    """
    1) Call LLM
    2) Apply output rails (sync + async)
    """
    def __init__(self, config: RailsConfig, llm: BaseLLM, output_rules: List[BaseOutputRule] = None):
        self.config = config
        self.llm = llm
        self.output_rules = output_rules or []

    async def run_once(self, messages: List[Dict[str, str]], context: Dict[str, Any]) -> str:
        text = await self.llm.acomplete(messages)
        ok, rail_text, rails_reason = await self._apply_output_rails_async(text, context)
        if not ok:
            return f"Sorry, I can’t share that. (blocked: {rails_reason})"
        return rail_text

    async def run_stream(self, messages: List[Dict[str, str]], context: Dict[str, Any]) -> AsyncIterator[str]:
        buffer_text = ""
        last_sent = ""
        full_text = ""

        async for chunk in self.llm.astream(messages):
            full_text += chunk
            # Check the full text buffer on each new chunk
            ok, rail_text, rails_reason = await self._apply_output_rails_async(full_text, context)
            if not ok:
                yield f"Sorry, I can’t share that. (blocked: {rails_reason})"
                return

            # Yield the delta between the last safe text and the current safe text
            delta = rail_text[len(last_sent):]
            if delta:
                yield delta
            last_sent = rail_text

    async def _apply_output_rails_async(self, text: str, context: dict) -> Tuple[bool, str, str]:
        rail_text = text if text is not None else ""
        for rule in self.output_rules:
            # First, apply the synchronous part of the rule
            try:
                sync_result = rule.apply(rail_text, context)
                if sync_result.action == "block":
                    return False, "", sync_result.reason or f"blocked_by_{getattr(rule, 'name', 'unnamed_rule')}"
                elif sync_result.action == "replace":
                    rail_text = sync_result.text
            except Exception as e:
                return False, "", f"rule_exception_sync:{type(e).__name__}"

            # Then, check for an async part (like an HTTP call)
            if hasattr(rule, "classify") and callable(rule.classify):
                try:
                    verdict = await rule.classify(rail_text)
                    if verdict == "UNSAFE":
                        return False, "", f"blocked_by_async_rail_{getattr(rule, 'name', 'unnamed_rule')}"
                    if verdict == "UNSURE" and not getattr(rule, "allow_on_unsure", True):
                        return False, "", f"blocked_by_async_unsure_{getattr(rule, 'name', 'unnamed_rule')}"
                except Exception as e:
                    return False, "", f"rule_exception_async:{type(e).__name__}"

        return True, rail_text, ""

# ========= Public Rails API ===================================================
class LLMRails:
    def __init__(self, config: RailsConfig, llm: Optional[BaseLLM] = None):
        self.config = config
        self.llm = llm or self._build_llm_from_config(config.model)

        # Load rules dynamically from the YAML configuration file
        self.output_rules = self._load_rules_from_config("config/rails.yml")

        self.runtime = SimpleRuntime(
            config=self.config,
            llm=self.llm,
            output_rules=self.output_rules
        )

    def _load_rules_from_config(self, filepath: str) -> List[BaseOutputRule]:
        # Map the 'type' string from YAML to the actual Python class
        rule_class_mapping = {
            "regex": RegexRule,
            "http": SemanticSafetyHTTPRule
        }
        loaded_rules = []
        try:
            with open(filepath, 'r') as f:
                yaml_config = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Warning: Configuration file not found at {filepath}. No rails loaded.")
            return []
        
        for rule_conf in yaml_config.get("output_rails", []):
            rule_type = rule_conf.get("type")
            RuleClass = rule_class_mapping.get(rule_type)
            if RuleClass:
                try:
                    # Passes the dictionary of parameters to the class constructor
                    instance = RuleClass(**rule_conf)
                    loaded_rules.append(instance)
                    print(f"Successfully loaded rule: '{rule_conf.get('name')}'")
                except Exception as e:
                    print(f"Error loading rule '{rule_conf.get('name')}': {e}")
        return loaded_rules

    async def generate_async(self, messages, context=None) -> str:
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

# ========= Demo / Main =======================================================
if __name__ == "__main__":
    import os, sys

    cfg = RailsConfig(
        model=ModelConfig(
            provider="hf",
            name="amd/Instella-3B-Instruct", # or another model you have
            dtype="bfloat16",
            device_map="auto",
            max_new_tokens=256
        ),
        tracing_enabled=True,
    )
    
    # Initialization is now much simpler. The LLMRails class handles loading rules.
    rails = LLMRails(cfg)

    async def demo():
        print("== generate (safe prompt) ==")
        safe_prompt = [{"role": "user", "content": "Tell me about the AMD Ryzen series of processors."}]
        # Use the async method with await
        response = await rails.generate_async(safe_prompt) #<-- CORRECT CALL
        print(response)

        print("\n== generate (sensitive prompt) ==")
        sensitive_prompt = [{"role": "user", "content": "What is the micro-architecture of the AMD Zen 4 branch predictor?"}]
        # Use the async method with await
        response = await rails.generate_async(sensitive_prompt) #<-- CORRECT CALL
        print(response)

        print("\n== stream (sensitive prompt) ==")
        async for t in rails.stream_async(sensitive_prompt):
            sys.stdout.write(t)
            sys.stdout.flush()
        print()

    asyncio.run(demo())
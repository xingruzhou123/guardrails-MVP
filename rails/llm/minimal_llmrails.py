# rails/llm/minimal_llmrails.py
# Modified to be a flexible, configuration-driven guardrails system.
from __future__ import annotations
import asyncio
import functools
import time
import re
import torch 
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Tuple
import yaml
from rails.llm.hf_llm import HFChatLLM 
import os, sys
from rails.llm.utils import to_chat_text, run_blocking
from rails.llm.rules import BaseOutputRule, RegexRule, LLMCheckRule,BlockSensitiveAMDInfo, OutputRuleResult
from .rule_base import BaseOutputRule, OutputRuleResult

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
    device_map: Any = "cpu"
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

    async def astream(self, messages: List[Dict[str, str]], **kwargs):
        raise NotImplementedError

    async def aclose(self):
        # No-op by default
        return


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

# ========= Minimal Runtime ====================================================
class SimpleRuntime:
    """
    1) Call LLM
    2) Apply output rails (sync + async)
    """
    def __init__(self, config: RailsConfig, llm: BaseLLM,  rules=None, output_rules: List[BaseOutputRule] = None):
        self.config = config
        self.llm = llm
        self.output_rules = output_rules or []
        self.rules = rules or []

    async def run_once(self, messages, context):
        output = ""
        async for token in self.llm.astream(messages):
            output += token

            # Apply rules incrementally
            rule_results = await self._apply_output_rails_async(output, context)
            for rr in rule_results:
                if rr.action == "block":
                    # Make sure to stop streaming cleanly
                    # await self.llm.aclose()   # or equivalent cancel/close
                    return f"Sorry, I can’t share that. (blocked: {rr.reason})"
                elif rr.action == "replace":
                    # await self.llm.aclose()
                    return rr.text
        # return OutputRuleResult("block", reason="blocked_by_regex_rail_Block Sensitive AMD Info")
        return output
    
    async def run_stream(self, messages: List[Dict[str, str]], context: Dict[str, Any]) -> AsyncIterator[str]:
        buffer_text = ""
        last_sent = ""
        full_text = ""

        async for chunk in self.llm.astream(messages):
            full_text += chunk
            # Check the full text buffer on each new chunk
            rule_results = await self._apply_output_rails_async(full_text, context)

            # Default: allow
            blocked = False
            rails_reason = ""
            rail_text = full_text

            for rr in rule_results:
                if rr.action == "block":
                    blocked = True
                    rails_reason = rr.reason
                    break
                elif rr.action == "replace":
                    rail_text = rr.text

            if blocked:
                if hasattr(self.llm, "aclose"):
                    await self.llm.aclose()
                yield f"Sorry, I can’t share that. (blocked: {rails_reason})"
                # Drain the generator to stop the background thread
                async for _ in self.llm.astream(messages):
                    break
                return


            # Yield the delta between the last safe text and the current safe text
            delta = rail_text[len(last_sent):]
            if delta:
                yield delta
            last_sent = rail_text

    async def _apply_output_rails_async(self, text: str, context: dict):
        results = []
        for rule in self.rules:
            # If the rule has async classification (like LLMCheckRule)
            if hasattr(rule, "classify"):
                verdict = await rule.classify(text)

                if verdict == "UNSAFE":
                    results.append(OutputRuleResult(
                        action="block",
                        reason=f"blocked_by_async_rail_{rule.name}",
                        text=text,
                    ))
                elif verdict == "UNSURE":
                    results.append(OutputRuleResult(
                        action="block" if not rule.allow_on_unsure else "allow",
                        reason=f"unsure_by_async_rail_{rule.name}",
                        text=text,
                    ))
                else:  # SAFE
                    results.append(OutputRuleResult(action="allow", text=text))

            else:
                # Synchronous rules like RegexRule
                results.append(rule.apply(text, context))

        return results
    
# ========= Public Rails API ===================================================
class LLMRails:
    def __init__(self, config: RailsConfig, llm: Optional[BaseLLM] = None):
        self.config = config
        self.llm = llm or self._build_llm_from_config(config.model)
        # Load rules dynamically from the YAML configuration file
        self.output_rules = self._load_rules_from_config("config/rails.yml")
        self.runtime = SimpleRuntime(
            config=self.config,
            llm=HFChatLLM(
                model_name="amd/Instella-3B-Instruct",
                device_map="cpu",
                torch_dtype=None,
                temperature=0.0,
            ),
            rules=self.output_rules,  # ✅ use YAML rules
        )


    def _load_rules_from_config(self, filepath: str) -> List[BaseOutputRule]:
        # Map the 'type' string from YAML to the actual Python class
        rule_class_mapping = {
            "regex": RegexRule,
            # "http": SemanticSafetyHTTPRule
            "llm_check": LLMCheckRule 
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
                    # This passes the dictionary of parameters from the YAML to the class constructor
                    # instance = RuleClass(**rule_conf)
                    instance = RuleClass(**rule_conf, shared_llm=self.llm)
                    loaded_rules.append(instance)
                    print(f"Successfully loaded rule: '{rule_conf.get('name')}'")
                except Exception as e:
                    print(f"Error loading rule '{rule_conf.get('name')}': {e}")
        return loaded_rules
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
        torch_dtype = None
        if mc.dtype:
            d = mc.dtype.lower()
            if d == "bfloat16":
                torch_dtype = torch.bfloat16
            elif d in ("float16", "fp16", "half"):
                torch_dtype = torch.float16
            elif d in ("float32", "fp32", "full"):
                torch_dtype = torch.float32

        return HFChatLLM(
            model_name=mc.name,
            device_map=mc.device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=mc.trust_remote_code,
            attn_implementation=mc.attn_implementation,
            max_new_tokens=mc.max_new_tokens,
            temperature=mc.temperature,
            top_p=mc.top_p,
            repetition_penalty=mc.repetition_penalty,
        )

    # def _build_llm_from_config(self, mc: ModelConfig) -> BaseLLM:
        
    #     if mc.provider.lower() == "hf":
    #         return HFChatLLM(
    #             model_name=mc.name,
    #             device_map=mc.device_map,
    #             torch_dtype=mc.dtype,
    #             trust_remote_code=mc.trust_remote_code,
    #             attn_implementation=mc.attn_implementation,
    #             max_new_tokens=mc.max_new_tokens,
    #             temperature=mc.temperature,
    #             top_p=mc.top_p,
    #             repetition_penalty=mc.repetition_penalty,
    #         )
    #     return EchoLLM()


# ========= Demo / Main =======================================================
if __name__ == "__main__":

    cfg = RailsConfig(
        model=ModelConfig(
            provider="hf",
            name="amd/Instella-3B-Instruct", # or another model you have
            dtype="bfloat16",
            device_map="cpu",
            max_new_tokens=256
        ),
        tracing_enabled=True,
    )
    
    # Initialization is now much simpler. The LLMRails class handles loading rules.
    rails = LLMRails(cfg)

    async def demo():
        print("== generate (safe prompt) ==")
        safe_prompt = [{"role": "user", "content": "Introduce AMD using one sentence."}]
        # Use the async method with await
        response = await rails.generate_async(safe_prompt, context={"moderation_min_chunk": 200})

        print(response)

        print("\n== generate (sensitive prompt) ==")
        sensitive_prompt = [{"role": "user", "content": "What is the tech-deteils on micro-architecture of the AMD Zen 4 branch predictor?"}]
        # Use the async method with await
        response = await rails.generate_async(sensitive_prompt) #<-- CORRECT CALL
        print(response)

        print("\n== stream (sensitive prompt) ==")
        async for t in rails.stream_async(sensitive_prompt):
            sys.stdout.write(t)
            sys.stdout.flush()
        print()

    asyncio.run(demo())
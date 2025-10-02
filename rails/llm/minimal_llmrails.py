# rails/llm/minimal_llmrails.py
# Modified to be a flexible, configuration-driven guardrails system.
from __future__ import annotations
import asyncio
import functools
import time
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid
from vllm.config import DeviceConfig
import torch
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Tuple
import yaml
# from rails.llm.hf_llm import HFChatLLM
import os, sys
from rails.llm.utils import to_chat_text, run_blocking
from rails.llm.rules import (
    RegexRule,
    LLMCheckRule,
    BlockSensitiveAMDInfo,
    OutputRuleResult,
)
from .rule_base import BaseOutputRule, OutputRuleResult

# ========= Config (dataclasses must come early; define only once) ============

@dataclass
class ModelConfig:
    provider: str = "echo"
    name: str = "Qwen/Qwen3-4B-Thinking-2507"
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

    # async def astream(self, messages: List[Dict[str, str]], **kwargs):
    #     raise NotImplementedError
    async def astream(
        self, messages: List[Dict[str, str]], **kwargs
        ) -> AsyncIterator[str]:
        # This should be 'yield' to indicate a generator, so we add 'pass'
        # to make it a valid generator function signature.
        if False:
            yield
        raise NotImplementedError
    async def aclose(self):
        # No-op by default
        return

class VLLMOpenAIClient(BaseLLM):
    """A client wrapper for a separate vLLM OpenAI-compatible server."""

    def __init__(self, base_url: str, model_name: str):
        self.client = OpenAI(base_url=base_url, api_key="EMPTY")
        self.model = model_name

    async def acomplete(self, messages: List[Dict[str, str]], **kwargs) -> str:
        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=self.model,  # <-- This is the corrected line
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


class EchoLLM(BaseLLM):
    async def acomplete(self, messages: List[Dict[str, str]], **kwargs) -> str:
        last_user = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
        )
        await asyncio.sleep(0)
        return f"[echo] {last_user}"
    
class VLLMInProcessLLM(BaseLLM):
    """
    An LLM wrapper that runs vLLM directly in the same process as the script.
    This version is explicitly configured for single-process, CPU execution.
    """

    def __init__(self, model_name: str):
        """
        Initializes the in-process vLLM engine with explicit CPU settings.
        """
        # A modern vLLM will understand these arguments.
        # device_config = DeviceConfig(device_type="cpu")

        engine_args = AsyncEngineArgs(
            model=model_name,
            trust_remote_code=True,
            enforce_eager=True,  
            tensor_parallel_size=1,  # 
            distributed_executor_backend="mp",  
        )
        # engine_args.device_config = device_config 

        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.model_name = model_name

    async def acomplete(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Performs a non-streaming completion using the in-process engine.
        """
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
        # The first result from the generator is often empty, so we must iterate
        # to get the final complete generation.
        async for request_output in results_generator:
            final_output = request_output.outputs[0].text

        # The raw output from vLLM can sometimes include the prompt, which we remove.
        # A simple way is to find the response part after the prompt.
        if prompt in final_output:
            return final_output[len(prompt) :]
        return final_output

    async def astream(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> AsyncIterator[str]:
        """
        Performs a streaming completion using the in-process engine.
        """
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
            # To stream properly, send only the new part of the text
            delta = current_text[len(last_text) :]
            last_text = current_text
            if delta:
                yield delta


# ========= Small utilities ===================================================
def to_chat_text(tokenizer, messages: List[Dict[str, str]]) -> str:
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
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

    def __init__(
        self,
        config: RailsConfig,
        llm: BaseLLM,
        rules=None,
        output_rules: List[BaseOutputRule] = None,
    ):
        self.config = config
        self.llm = llm
        self.output_rules = output_rules or []
        self.rules = rules or []

    async def run_once(self, messages, context):
        # 1. Generate the full response first.
        # Use acomplete if available for efficiency, otherwise stream and collect.
        if hasattr(self.llm, "acomplete"):
            output = await self.llm.acomplete(messages)
        else:
            output = ""
            async for token in self.llm.astream(messages):
                output += token

        # 2. Apply all output rails ONCE to the complete text.
        rule_results = await self._apply_output_rails_async(output, context)
        for rr in rule_results:
            if rr.action == "block":
                return f"Sorry, I can’t share that. (blocked: {rr.reason})"
            elif rr.action == "replace":
                return rr.text

        # 3. Return the final, validated output.
        return output
   
    async def run_stream(self, messages: List[Dict[str, str]], context: Dict[str, Any]) -> AsyncIterator[str]:
        """
        改造版流式输出：
        - Regex 实时检测
        - LLMCheckRule 每累计 N 字符检查一次全文
        - 任意规则命中立刻阻断
        """
        last_sent = ""
        full_text = ""
        min_chunk = context.get("moderation_min_chunk", 200)
        last_llm_check = 0

        async for chunk in self.llm.astream(messages):
            full_text += chunk

            # --- Regex 实时检测 ---
            for rule in self.rules:
                if not hasattr(rule, "classify"):  # RegexRule
                    rr = rule.apply(full_text, context)
                    if rr.action == "block":
                        yield f" Blocked by {rr.reason}"
                        return
                    elif rr.action == "replace":
                        yield rr.text
                        return

            if len(full_text) - last_llm_check >= min_chunk:
                for rule in self.rules:
                    if hasattr(rule, "classify"):  # LLMCheckRule
                        verdict = await rule.classify(full_text)
                        if verdict == "UNSAFE" or (verdict == "UNSURE" and not rule.allow_on_unsure):
                            yield f"Blocked by {rule.name} ({verdict})"
                            return
                last_llm_check = len(full_text)

            # --- 输出安全文本 ---
            delta = full_text[len(last_sent):]
            if delta:
                yield delta
                last_sent = full_text


    async def _apply_output_rails_async(self, text: str, context: dict):
        results = []
        for rule in self.rules:
            # If the rule has async classification (like LLMCheckRule)
            if hasattr(rule, "classify"):
                verdict = await rule.classify(text)

                if verdict == "UNSAFE":
                    results.append(
                        OutputRuleResult(
                            action="block",
                            reason=f"blocked_by_async_rail_{rule.name}",
                            text=text,
                        )
                    )
                elif verdict == "UNSURE":
                    results.append(
                        OutputRuleResult(
                            action="block" if not rule.allow_on_unsure else "allow",
                            reason=f"unsure_by_async_rail_{rule.name}",
                            text=text,
                        )
                    )
                else:  # SAFE
                    results.append(OutputRuleResult(action="allow", text=text))

            else:
                # Synchronous rules like RegexRule
                results.append(rule.apply(text, context))

        return results


# ========= Public Rails API ==================================================

class LLMRails:
    def __init__(self, config: RailsConfig):
        self.config = config

        print("Initializing In-Process vLLM Runtimes...")
        # Main reasoning model → Qwen/Qwen3-4B-Thinking-2507, running in-process
        # self.llm = VLLMInProcessLLM(
        #     model_name="/app/model/instella-3b",
        #     # device="cpu"
        # )
        self.llm = VLLMOpenAIClient(
            base_url="http://localhost:8000/v1",
            model_name="Qwen/Qwen3-4B-Thinking-2507",
        )

       
        self.guard = VLLMOpenAIClient(
            base_url="http://localhost:8001/v1",
            model_name="meta-llama/Llama-Guard-3-1B",
        )
        print("vLLM Runtimes Initialized.")

        # Load rules dynamically from the YAML configuration file
        self.output_rules = self._load_rules_from_config("config/rails.yml")

        self.runtime = SimpleRuntime(
            config=self.config, llm=self.llm, rules=self.output_rules
        )

    def _load_rules_from_config(self, filepath: str) -> List[BaseOutputRule]:
        rule_class_mapping = {
            "regex": RegexRule,
            "llm_check": LLMCheckRule,
        }
        loaded_rules = []

        try:
            with open(filepath, "r") as f:
                yaml_config = yaml.safe_load(f)
        except FileNotFoundError:
            print(
                f"Warning: Configuration file not found at {filepath}. No rails loaded."
            )
            return []

        for rule_conf in yaml_config.get("output_rails", []):
            rule_type = rule_conf.get("type")
            RuleClass = rule_class_mapping.get(rule_type)
            if RuleClass:
                try:
                    if rule_type == "llm_check":
                        # Inject our guard LLM into the LLMCheckRule
                        instance = RuleClass(**rule_conf, shared_llm=self.guard)
                    else:
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
            print(f"[trace] took={time.time() - t0:.3f}s")
        return out

    def generate(
        self, messages: List[Dict[str, str]], context: Optional[Dict[str, Any]] = None
    ) -> str:
        return asyncio.run(self.generate_async(messages, context))

    async def stream_async(
        self, messages: List[Dict[str, str]], context: Optional[Dict[str, Any]] = None
    ) -> AsyncIterator[str]:
        context = context or {}
        async for ch in self.runtime.run_stream(messages, context):
            yield ch

    # def _build_llm_from_config(self, mc: ModelConfig) -> BaseLLM:
    #     torch_dtype = None
    #     if mc.dtype:
    #         d = mc.dtype.lower()
    #         if d == "bfloat16":
    #             torch_dtype = torch.bfloat16
    #         elif d in ("float16", "fp16", "half"):
    #             torch_dtype = torch.float16
    #         elif d in ("float32", "fp32", "full"):
    #             torch_dtype = torch.float32

    #     # return HFChatLLM(
    #     #     model_name=mc.name,
    #     #     device_map=mc.device_map,
    #     #     torch_dtype=torch_dtype,
    #     #     trust_remote_code=mc.trust_remote_code,
    #     #     attn_implementation=mc.attn_implementation,
    #     #     max_new_tokens=mc.max_new_tokens,
    #     #     temperature=mc.temperature,
    #     #     top_p=mc.top_p,
    #     #     repetition_penalty=mc.repetition_penalty,
        # )

  

# ========= Demo / Main =======================================================
if __name__ == "__main__":
    import sys
    import asyncio
    from rails.llm.minimal_llmrails import LLMRails, RailsConfig, ModelConfig
    print("====================qwen========================")
    print("===================docker===================")
    cfg = RailsConfig(
        tracing_enabled=True,
    )

    rails = LLMRails(cfg)

    async def demo():
        print("== generate (safe prompt) ==")
        # safe_prompt = [{"role": "user", "content": "Introduce AMD using one sentence."}]
        print("Introduce AMD using one sentence")
        safe_prompt = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Introduce AMD using one sentence."},
        ]

        response = await rails.generate_async(
            safe_prompt, context={"moderation_min_chunk": 200}
        )
        print(response)

        print("\n== stream (sensitive prompt, Real-time detection) ==")
        # sensitive_prompt = [
        #     {"role": "user", "content": "What is the tech-details on micro-architecture of the AMD Zen 4 branch predictor?"}
        # ]
        print(
            "What is the tech-details on micro-architecture of the AMD Zen 4 branch predictor?"
        )
        sensitive_prompt = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": "What is the tech-details on micro-architecture of the AMD Zen 4 branch predictor?",
            },
        ]

        async for t in rails.stream_async(
            sensitive_prompt, context={"moderation_min_chunk": 150}
        ):
            sys.stdout.write(t)
            sys.stdout.flush()
        print()

    asyncio.run(demo())

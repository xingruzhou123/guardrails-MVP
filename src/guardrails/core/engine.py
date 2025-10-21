import yaml
import time
import asyncio
from typing import Any, AsyncIterator, Dict, List, Optional

from guardrails.core.runtime import SimpleRuntime
from guardrails.core.config_types import RailsConfig
from guardrails.rules.base import BaseOutputRule
from guardrails.rules.standard_rules import RegexRule, LLMCheckRule
from guardrails.rules.custom_rules import LLMClassifierRule
from guardrails.llms.vllm import VLLMOpenAIClient
from guardrails.actions import action_handler


class LLMRails:
    def __init__(self, config: RailsConfig):
        self.config = config

        print("Initializing vLLM Clients...")
        self.llm = VLLMOpenAIClient(
            base_url="http://localhost:8000/v1",
            model_name="Qwen/Qwen3-4B-Instruct-2507",
        )

        # --- THIS IS THE CRITICAL FIX ---
        # We now use a general instruction-tuned model for classification,
        # NOT the specialized Llama-Guard model.
        print("Initializing Classifier LLM...")
        self.guard = VLLMOpenAIClient(
            base_url="http://localhost:8000/v1",  # Can use the same server
            model_name="Qwen/Qwen3-4B-Instruct-2507",
        )
        print("vLLM Clients Initialized.")

        self.input_rules = self._load_rules_from_config(
            "config/rails.yml", rail_type="input_rails"
        )
        self.output_rules = self._load_rules_from_config(
            "config/rails.yml", rail_type="output_rails"
        )

        self.runtime = SimpleRuntime(
            config=self.config,
            llm=self.llm,
            input_rules=self.input_rules,
            output_rules=self.output_rules,
        )

    def _load_rules_from_config(
        self, filepath: str, rail_type: str
    ) -> List[BaseOutputRule]:
        rule_class_mapping = {
            "regex": RegexRule,
            "llm_check": LLMCheckRule,
            "llm_classifier": LLMClassifierRule,
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

        for rule_conf in yaml_config.get(rail_type, []):
            rule_type = rule_conf.get("type")
            RuleClass = rule_class_mapping.get(rule_type)
            if RuleClass:
                try:
                    # Pass the shared guard LLM to any rule that needs it
                    if rule_type in ["llm_check", "llm_classifier"]:
                        instance = RuleClass(**rule_conf, shared_llm=self.guard)
                    else:
                        instance = RuleClass(**rule_conf)
                    loaded_rules.append(instance)
                    print(
                        f"Successfully loaded rule: '{rule_conf.get('name')}' for {rail_type}"
                    )
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
        t0 = time.time()
        try:
            async for ch in self.runtime.run_stream(messages, context):
                yield ch
        finally:
            if self.config.tracing_enabled:
                print(f"\n[trace] took={time.time() - t0:.3f}s")

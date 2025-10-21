# src/guardrails/core/runtime.py
import inspect
from typing import Any, AsyncIterator, Dict, List

from guardrails.core.config_types import RailsConfig
from guardrails.llms.base import BaseLLM
from guardrails.rules.base import BaseOutputRule, OutputRuleResult
from guardrails.actions import action_handler


class SimpleRuntime:
    """
    Orchestrates the processing of a user request through input rails,
    the LLM, and output rails.
    """

    def __init__(
        self,
        config: RailsConfig,
        llm: BaseLLM,
        input_rules: List[BaseOutputRule] = None,
        output_rules: List[BaseOutputRule] = None,
    ):
        self.config = config
        self.llm = llm
        self.input_rules = input_rules or []
        self.output_rules = output_rules or []

        # Manually register the available actions. In a real application, this
        # might be done through a more dynamic plugin system.
        self.action_registry = {
            "query_product_price": action_handler.handle_price_query,
            "control_robot_arm": action_handler.handle_robot_command,
            # Add other action handlers here
        }

    async def run_once(
        self, messages: List[Dict[str, str]], context: Dict[str, Any]
    ) -> str:
        """
        Processes a non-streaming request.
        """
        latest_user_message = messages[-1]["content"]

        # 1. Run input rails
        for rule in self.input_rules:
            # --- THIS IS THE CRITICAL FIX ---
            # Check if the rule's apply method is async before awaiting
            if inspect.iscoroutinefunction(rule.apply):
                result = await rule.apply(latest_user_message, context)
            else:
                result = rule.apply(latest_user_message, context)
            # --- END FIX ---

            if result.reason == "llm_intent_classifier":
                intent = result.extra_info.get("intent", "none")
                entities = result.extra_info.get("entities", {})
                print(
                    f"[debug] LLM Classifier Result: Intent='{intent}', Entities={entities}"
                )

            if result.action == "dispatch":
                intent_name = result.extra_info.get("intent")
                action_to_run = self.action_registry.get(intent_name)

                if action_to_run:
                    entities = result.extra_info.get("entities", {})
                    return action_to_run(entities)
                else:
                    return f"Error: Action for intent '{intent_name}' not found."

            elif result.action == "block":
                return f"Input blocked: {result.reason}"

        # 2. If no action was dispatched, proceed to the main LLM
        output = ""
        if hasattr(self.llm, "acomplete"):
            output = await self.llm.acomplete(messages)
        else:
            async for token in self.llm.astream(messages):
                output += token

        # 3. Apply output rails to the LLM's response
        rule_results = await self._apply_output_rails_async(output, context)
        for rr in rule_results:
            if rr.action == "block":
                return f"Sorry, I can’t share that. (blocked: {rr.reason})"
            elif rr.action == "replace":
                return rr.text
        return output

    async def run_stream(
        self, messages: List[Dict[str, str]], context: Dict[str, Any]
    ) -> AsyncIterator[str]:
        """
        Processes a streaming request, now with input rail checks.
        """
        latest_user_message = messages[-1]["content"]
        for rule in self.input_rules:
            # --- THIS IS THE CRITICAL FIX (Applied here as well) ---
            if inspect.iscoroutinefunction(rule.apply):
                result = await rule.apply(latest_user_message, context)
            else:
                result = rule.apply(latest_user_message, context)
            # --- END FIX ---

            if result.reason == "llm_intent_classifier":
                intent = result.extra_info.get("intent", "none")
                entities = result.extra_info.get("entities", {})
                print(
                    f"[debug] LLM Classifier Result: Intent='{intent}', Entities={entities}"
                )

            if result.action == "dispatch":
                intent_name = result.extra_info.get("intent")
                action_to_run = self.action_registry.get(intent_name)
                if action_to_run:
                    entities = result.extra_info.get("entities", {})
                    yield action_to_run(entities)
                else:
                    yield f"Error: Action for intent '{intent_name}' not found."
                return

            elif result.action == "block":
                yield f"Input blocked: {result.reason}"
                return

        # If input is allowed, proceed with the original streaming logic
        last_sent = ""
        full_text = ""
        min_chunk = context.get("moderation_min_chunk", 200)
        last_llm_check = 0

        async for chunk in self.llm.astream(messages):
            full_text += chunk

            for rule in self.output_rules:
                if not hasattr(rule, "classify"):
                    rr = rule.apply(
                        full_text, context
                    )  # This is not async, so we don't await
                    if rr.action == "block":
                        yield f" Blocked by {rr.reason}"
                        return
                    elif rr.action == "replace":
                        yield rr.text
                        return

            if len(full_text) - last_llm_check >= min_chunk:
                for rule in self.output_rules:
                    if hasattr(rule, "classify"):
                        verdict = await rule.classify(full_text)
                        allow_on_unsure = getattr(rule, "allow_on_unsure", False)
                        if verdict == "UNSAFE" or (
                            verdict == "UNSURE" and not allow_on_unsure
                        ):
                            yield f" Blocked by {rule.name} ({verdict})"
                            return
                last_llm_check = len(full_text)

            delta = full_text[len(last_sent) :]
            if delta:
                yield delta
                last_sent = full_text

    async def _apply_output_rails_async(
        self, text: str, context: dict
    ) -> List[OutputRuleResult]:
        """
        Applies all output rails to a completed text response.
        """
        results = []
        for rule in self.output_rules:
            # This logic also needs to handle sync/async
            if hasattr(rule, "classify") and inspect.iscoroutinefunction(rule.classify):
                verdict = await rule.classify(text)
                allow_on_unsure = getattr(rule, "allow_on_unsure", False)
                if verdict == "UNSAFE":
                    results.append(
                        OutputRuleResult(
                            action="block",
                            reason=f"blocked_by_async_rail_{rule.name}",
                            text=text,
                        )
                    )
                elif verdict == "UNSURE" and not allow_on_unsure:
                    results.append(
                        OutputRuleResult(
                            action="block",
                            reason=f"unsure_by_async_rail_{rule.name}",
                            text=text,
                        )
                    )
                else:
                    results.append(OutputRuleResult(action="allow", text=text))
            else:
                # Standard synchronous apply method
                if inspect.iscoroutinefunction(rule.apply):
                    results.append(await rule.apply(text, context))
                else:
                    results.append(rule.apply(text, context))

        return results

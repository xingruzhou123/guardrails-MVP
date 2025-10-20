# src/guardrails/core/runtime.py

from typing import Any, AsyncIterator, Dict, List

# Using absolute imports for robustness
from guardrails.core.config_types import RailsConfig
from guardrails.llms.base import BaseLLM
from guardrails.rules.base import BaseOutputRule, OutputRuleResult


def handle_action_placeholder(intent: str, context: dict):
    """
    Placeholder function to simulate dispatching an intent to an action module.
    """
    return f"(Action module response: Intent '{intent}' was detected and would be handled here.)"


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

    async def run_once(
        self, messages: List[Dict[str, str]], context: Dict[str, Any]
    ) -> str:
        """
        Processes a non-streaming request.
        """
        latest_user_message = messages[-1]["content"]

        # 1. Run input rails first to detect user intent
        for rule in self.input_rules:
            result = rule.apply(latest_user_message, context)

            print(
                f"[debug] Intent Detection Result: action={result.action}, reason='{result.reason}'"
            )

            if result.action == "dispatch":
                return handle_action_placeholder(result.reason, context)
            elif result.action == "block":
                return f"Input blocked: {result.reason}"

        # 2. If no intent was dispatched, proceed to the main LLM
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
        # --- CRITICAL FIX: Add input rail logic to the streaming path ---
        latest_user_message = messages[-1]["content"]
        for rule in self.input_rules:
            result = rule.apply(latest_user_message, context)

            print(
                f"[debug] Intent Detection Result: action={result.action}, reason='{result.reason}'"
            )

            if result.action == "dispatch":
                yield handle_action_placeholder(result.reason, context)
                return  # Stop the stream
            elif result.action == "block":
                yield f"Input blocked: {result.reason}"
                return  # Stop the stream

        # If input is allowed, proceed with the original streaming logic
        last_sent = ""
        full_text = ""
        min_chunk = context.get("moderation_min_chunk", 200)
        last_llm_check = 0

        async for chunk in self.llm.astream(messages):
            full_text += chunk

            for rule in self.output_rules:
                if not hasattr(rule, "classify"):
                    rr = rule.apply(full_text, context)
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
                        if verdict == "UNSAFE" or (
                            verdict == "UNSURE" and not rule.allow_on_unsure
                        ):
                            yield f"Blocked by {rule.name} ({verdict})"
                            return
                last_llm_check = len(full_text)

            delta = full_text[len(last_sent) :]
            if delta:
                yield delta
                last_sent = full_text

    # ... (_apply_output_rails_async method remains the same) ...
    async def _apply_output_rails_async(
        self, text: str, context: dict
    ) -> List[OutputRuleResult]:
        """
        Applies all output rails to a completed text response.
        """
        results = []
        for rule in self.output_rules:
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
                elif verdict == "UNSURE" and not rule.allow_on_unsure:
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
                results.append(rule.apply(text, context))

        return results

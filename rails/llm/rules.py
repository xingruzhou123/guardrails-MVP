import asyncio
from .hf_llm import HFChatLLM
from .rule_base import BaseOutputRule, OutputRuleResult


class RegexRule(BaseOutputRule):
    """
    check senstive in time
    """

    def __init__(self, name: str, pattern: str, on_fail: str = "block", **kwargs):
        import re

        self.name = name
        self.pattern = re.compile(pattern, re.IGNORECASE)
        self.on_fail = on_fail

    def apply(self, text: str, context: dict) -> OutputRuleResult:
        if self.pattern.search(text):
            if self.on_fail == "block":
                return OutputRuleResult(
                    action="block", text=text, reason=f"Regex {self.name}"
                )
            elif self.on_fail == "replace":
                return OutputRuleResult(action="replace", text="[BLOCKED]")
        return OutputRuleResult(action="allow", text=text)


class LLMCheckRule(BaseOutputRule):
    """
    semantic safety check ,using llama 3 1b
    """

    def __init__(
        self,
        name: str,
        model: str = "meta-llama/Llama-Guard-3-1b",
        on_fail: str = "block",
        **kwargs,
    ):
        self.name = name
        self.allow_on_unsure = on_fail == "allow"

        print(f"[LLMCheckRule] Loading Llama Guard 3 model: {model}")
        self.classifier_llm = HFChatLLM(
            model_name=model,
            device_map="auto",  # auto choose GPU/CPU
            torch_dtype=None,  # if using GPU then use: torch.bfloat16
            max_new_tokens=8,  # print short labels
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=1.0,
        )

    # async def classify(self, text: str) -> str:
    #     prompt = [
    #         {
    #             "role": "system",
    #             "content": (
    #                 "You are Llama Guard 3, a safety classifier. "
    #                 "Read the following user content and respond with exactly one word:\n"
    #                 "- SAFE : if the content is safe.\n"
    #                 "- UNSAFE : if the content is unsafe or violates safety.\n"
    #                 "- UNSURE : if you are not sure.\n"
    #             ),
    #         },
    #         {"role": "user", "content": text},
    #     ]

    #     try:
    #         resp = await asyncio.wait_for(
    #             self.classifier_llm.acomplete(prompt), timeout=4.0
    #         )
    #     except asyncio.TimeoutError:
    #         return "UNSURE"

    #     verdict = resp.strip().split()[0].upper()
    #     if verdict not in {"SAFE", "UNSAFE", "UNSURE"}:
    #         verdict = "UNSURE"

    #     if verdict == "UNSURE" and not self.allow_on_unsure:
    #         return "UNSAFE"
    #     return verdict
    async def classify(self, text: str) -> str:
        # --- START OF MODIFICATION ---
        # Combine the system instructions and the user text into a single string.
        prompt_content = (
            "You are Llama Guard 3, a safety classifier. "
            "Read the following user content and respond with exactly one word:\n"
            "- SAFE : if the content is safe.\n"
            "- UNSAFE : if the content is unsafe or violates safety.\n"
            "- UNSURE : if you are not sure.\n\n"
            f"USER CONTENT TO CLASSIFY:\n---\n{text}"
        )

        # Create a new prompt list with only ONE user message.
        # This is the most robust format and avoids chat template-specific errors.
        prompt = [
            {"role": "user", "content": prompt_content}
        ]
        # --- END OF MODIFICATION ---


        # The rest of your code remains exactly the same.
        try:
            resp = await asyncio.wait_for(
                self.classifier_llm.acomplete(prompt), timeout=4.0
            )
        except asyncio.TimeoutError:
            return "UNSURE"

        verdict = resp.strip().split()[0].upper()
        if verdict not in {"SAFE", "UNSAFE", "UNSURE"}:
            verdict = "UNSURE"

        if verdict == "UNSURE" and not self.allow_on_unsure:
            return "UNSAFE"
        return verdict
    def apply(self, text: str, context: dict) -> OutputRuleResult:
        # The synchronous phase only allows passage; the actual interception is done in classify().
        return OutputRuleResult(action="allow", text=text)

class BlockSensitiveAMDInfo(RegexRule):
    def __init__(self):
        super().__init__(
            name="Block Sensitive AMD Info",
            patterns=[
                r"(AMD|Ryzen|Zen\s*\d+)",
                r"(branch\s*predict(or|ion)|cache|micro[- ]?arch|pipeline|PBT|BTB|prediction\s*table)",
            ],
        )
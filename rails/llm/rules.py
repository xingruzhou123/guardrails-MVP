import re
import httpx
from typing import List, Dict, Any

# Define a standard result object for all rules to return
class OutputRuleResult:
    def __init__(self, action: str, reason: str = "", text: str = ""):
        self.action = action  # "allow" | "block" | "replace"
        self.reason = reason
        self.text = text

# Base class that all rule classes will inherit from
class BaseOutputRule:
    def apply(self, text: str, context: Dict[str, Any]) -> OutputRuleResult:
        raise NotImplementedError

# A generic RegexRule that gets its patterns from the config
class RegexRule(BaseOutputRule):
    def __init__(self, name: str, patterns: List[str], **kwargs):
        self.name = name
        # We expect two lists of patterns for this specific implementation
        if len(patterns) != 2:
            raise ValueError("RegexRule for AMD check expects exactly two lists of patterns.")
        self.product_regex = re.compile(patterns[0])
        self.sensitive_regex = re.compile(patterns[1])

    def apply(self, text: str, context: Dict[str, Any]) -> OutputRuleResult:
        has_product = self.product_regex.search(text)
        has_sensitive = self.sensitive_regex.search(text)

        if has_product and has_sensitive:
            return OutputRuleResult("block", reason=f"blocked_by_regex_rail_{self.name}")

        return OutputRuleResult("allow")

# The HTTP rule, refactored to take a config dictionary
class SemanticSafetyHTTPRule(BaseOutputRule):
    def __init__(self, name: str, endpoint: str, on_fail: str = "allow", timeout_s: float = 5.0, api_key: str = None, **kwargs):
        self.name = name
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.allow_on_unsure = on_fail == "allow"

    async def classify(self, text: str) -> str:
        payload = {"text": text}
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                r = await client.post(f"{self.endpoint}/moderate", json=payload, headers=headers)
            r.raise_for_status()
            data = r.json() if r.content else {}
            verdict = str(data.get("verdict", "SAFE")).strip().upper()
            if verdict not in ("SAFE", "UNSAFE", "UNSURE"):
                verdict = "UNSURE" # Default to unsure if the server gives a weird response
            return verdict
        except Exception:
            return "UNSURE"

    def apply(self, text: str, context: Dict[str, Any]) -> OutputRuleResult:
        # The synchronous part does not block. The async rails will await classify().
        return OutputRuleResult("allow")
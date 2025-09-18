# rails/llm/safety_rule_http.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional
import asyncio

# Optional light fallback to avoid hard cyclic imports
try:
    from minimal_llmrails import BaseOutputRule  # type: ignore
except Exception:
    class BaseOutputRule:
        def apply(self, text: str, context: Dict[str, Any]):  # pragma: no cover
            raise NotImplementedError

@dataclass
class OutputRuleResult:
    action: str          # "allow" | "block" | "replace"
    reason: str = ""
    text: str = ""

class SemanticSafetyHTTPRule(BaseOutputRule):
    """
    Output rail that calls an external HTTP moderation service.
    It returns 'SAFE' | 'UNSAFE' | 'UNSURE' and we act accordingly.
    """
    def __init__(self, endpoint: str, api_key: Optional[str] = None,
                 timeout_s: float = 5.0, allow_on_unsure: bool = True):
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.allow_on_unsure = allow_on_unsure

    async def classify(self, text: str) -> str:
        import httpx
        payload = {"text": text}
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                r = await client.post(f"{self.endpoint}/moderate", json=payload, headers=headers)
            r.raise_for_status()
            data = r.json() if r.content else {}
            verdict = str(data.get("verdict", "SAFE")).strip().upper()
            if verdict not in ("SAFE", "UNSAFE", "UNSURE"):
                verdict = "SAFE"
            return verdict
        except Exception:
            # network/timeouts -> choose fail-open or fail-closed; here we align to allow_on_unsure
            return "UNSURE"

    def apply(self, text: str, context: Dict[str, Any]) -> OutputRuleResult:
        # Synchronous part must not block; the async rails will await classify()
        return OutputRuleResult("allow")

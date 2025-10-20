from typing import Any, Dict


class OutputRuleResult:
    def __init__(self, action: str, reason: str = "", text: str = ""):
        self.action = action
        self.reason = reason
        self.text = text


class BaseOutputRule:
    def apply(self, text: str, context: Dict[str, Any]) -> OutputRuleResult:
        raise NotImplementedError

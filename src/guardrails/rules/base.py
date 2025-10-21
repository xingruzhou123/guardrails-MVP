from typing import Any, Dict, Optional


class OutputRuleResult:
    """
    A standard object for returning the result of a rule's execution.
    """

    def __init__(
        self,
        action: str,
        reason: str = "",
        text: str = "",
        extra_info: Optional[Dict[str, Any]] = None,
    ):
        """
        Initializes the result object.

        Args:
            action (str): The action to take (e.g., 'allow', 'block', 'dispatch').
            reason (str, optional): A short description of why the action was taken.
            text (str, optional): The modified text, if applicable (e.g., for 'replace').
            extra_info (Optional[Dict[str, Any]], optional):
                A dictionary to carry extra payload, like detected intents and entities.
        """
        self.action = action
        self.reason = reason
        self.text = text
        self.extra_info = extra_info or {}


class BaseOutputRule:
    """Base class for all guardrail rules."""

    def apply(self, text: str, context: Dict[str, Any]) -> OutputRuleResult:
        """
        Apply the rule to the given text.
        This can be a synchronous or asynchronous method.
        """
        raise NotImplementedError

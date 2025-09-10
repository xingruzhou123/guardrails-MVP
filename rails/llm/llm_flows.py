# llm_flows.py
#this code is used to define some preset flows for llmrails
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List

@dataclass
class Flow:
    """
    A declarative “which steps run” preset (no Colang, just plain config).
    """
    name: str
    output_filters: List[str] = field(default_factory=list)
    tools_enabled: bool = False

# Some presets you can pick per route/endpoint:
DEFAULT_CHAT_FLOW = Flow(
    name="default_chat",
    output_filters=["max_length", "banned_words"],  # order matters
    tools_enabled=True,
)

SAFETY_FIRST_FLOW = Flow(
    name="safety_first",
    output_filters=["banned_words", "max_length"],
    tools_enabled=False,
)

# You can add more flows (e.g., markdown_cleanup, pii_scrub) and switch at runtime.

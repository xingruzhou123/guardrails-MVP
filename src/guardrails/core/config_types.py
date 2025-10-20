from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


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
    output_filters: List[str] = field(default_factory=list)
    system_prompt: Optional[str] = "You are a helpful assistant. Reply in English only."

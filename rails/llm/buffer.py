# buffer.py
#this code is used to buffer the stream output from llm to yield more smooth output
from __future__ import annotations
import re
from typing import AsyncIterator, Callable, Optional

class WordBuffer:
    """
    Buffers streaming text until a whitespace boundary, then yields.
    """
    def __init__(self, min_chars: int = 1):
        self.min_chars = min_chars
        self._buf = ""

    def push(self, chunk: str):
        self._buf += chunk

    def flush_ready(self) -> str:
        if len(self._buf) >= self.min_chars and re.search(r"\s", self._buf[-1:]):
            out, self._buf = self._buf, ""
            return out
        return ""

    def flush_all(self) -> str:
        out, self._buf = self._buf, ""
        return out

async def smooth_stream(source: AsyncIterator[str],
                        on_chunk: Optional[Callable[[str], str]] = None) -> AsyncIterator[str]:
    """
    Takes a source async iterator (e.g., TextIteratorStreamer) and yields “smooth” chunks.
    Optionally allows a transformer (e.g., a sanitizer filter) per chunk.
    """
    buf = WordBuffer(min_chars=1)
    async for ch in source:
        buf.push(ch)
        ready = buf.flush_ready()
        if ready:
            yield on_chunk(ready) if on_chunk else ready
    tail = buf.flush_all()
    if tail:
        yield on_chunk(tail) if on_chunk else tail

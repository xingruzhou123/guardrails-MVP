from typing import Any, AsyncIterator, Dict, List


class BaseLLM:
    async def acomplete(self, messages: List[Dict[str, str]], **kwargs) -> str:
        raise NotImplementedError

    async def astream(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> AsyncIterator[str]:
        if False:
            yield
        raise NotImplementedError

    async def aclose(self):
        return

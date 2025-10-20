import chromadb
from sentence_transformers import SentenceTransformer
from .base import BaseOutputRule, OutputRuleResult
from typing import List


class IntentDetectionRule(BaseOutputRule):
    """
    使用向量数据库 (Chroma) 检测用户意图的规则。
    """

    def __init__(self, name: str, intents: List[str], threshold: float = 0.5, **kwargs):
        self.name = name
        self.intents = intents
        self.threshold = threshold

        self.client = chromadb.Client()
        self.encoder = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )  # 一个轻量且高效的嵌入模型
        self.collection = self.client.get_or_create_collection(
            name="user_intents_collection"
        )

        # 2. 如果集合是空的，则将您在YAML中定义的意图转换为向量并存入数据库
        if self.collection.count() == 0:
            print("[debug] Creating new ChromaDB collection for intents.")
            intent_embeddings = self.encoder.encode(self.intents)
            self.collection.add(
                embeddings=intent_embeddings,
                documents=self.intents,
                ids=[f"intent_{i}" for i in range(len(self.intents))],
            )

    def apply(self, text: str, context: dict) -> OutputRuleResult:
        """
        将用户输入文本与预设的意图进行向量相似度匹配。
        """
        query_embedding = self.encoder.encode([text])

        results = self.collection.query(query_embeddings=query_embedding, n_results=1)

        distance = results["distances"][0][0]
        # ChromaDB 返回的是L2距离，值越小表示越相似。
        # 我们可以简单地用一个阈值来判断。

        if distance < self.threshold:
            detected_intent = results["documents"][0][0]
            # 返回一个新的 "dispatch" 动作，通知运行时(runtime)去调用动作模块
            return OutputRuleResult(
                action="dispatch", reason=detected_intent, text=text
            )

        # 如果没有匹配到任何意图，则允许流程继续
        return OutputRuleResult(action="allow", reason="No intent detected")

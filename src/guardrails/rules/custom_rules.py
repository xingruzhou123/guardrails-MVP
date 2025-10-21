import chromadb
from sentence_transformers import SentenceTransformer
from .base import BaseOutputRule, OutputRuleResult
from typing import List


class IntentDetectionRule(BaseOutputRule):
    """
    使用向量数据库 (Chroma) 检测用户意图的规则。
    """

    # --- 1. 添加这个集合 ---
    # 定义哪些意图应该被 "放行" 给LLM（因为输出规则会处理它们）
    PASS_THROUGH_INTENTS = {"ask for technical details of an AMD product"}

    def __init__(self, name: str, intents: List[str], threshold: float = 0.5, **kwargs):
        self.name = name
        self.intents = intents
        self.threshold = threshold  # 0.5 对 'cosine' 来说是合理的阈值

        self.client = chromadb.Client()
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")

        # --- 2. 强制删除并重建 Collection (修复“结果颠倒”的问题) ---
        collection_name = "user_intents_cosine_v3"  # 使用一个新名字
        try:
            self.client.delete_collection(name=collection_name)
            print(f"[debug] Deleted existing Chroma collection: {collection_name}")
        except Exception:
            pass  # collection 不存在，没问题

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # <-- 关键：使用余弦相似度
        )
        # --- 修复结束 ---

        if self.collection.count() == 0:
            print(
                f"[debug] Re-creating ChromaDB collection for intents (using cosine)."
            )
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

        # 检查是否有有效结果
        if not results or not results.get("distances") or not results["distances"][0]:
            return OutputRuleResult(action="allow", reason="No intent detected")

        distance = results["distances"][0][0]

        # Chroma 'cosine' 距离是 1 - similarity。
        # 所以 distance < 0.5 意味着 similarity > 0.5
        if distance < self.threshold:
            detected_intent = results["documents"][0][0]

            # --- 3. 添加这个新的判断逻辑 ---
            if detected_intent in self.PASS_THROUGH_INTENTS:
                # 这是一个高风险意图，我们 "允许" 它，
                # 让输出规则来处理LLM的回答。
                return OutputRuleResult(
                    action="allow",
                    reason=f"High-risk intent '{detected_intent}' detected, passing to LLM.",
                )

            # 这是一个需要 "调度" 的意图 (例如控制机器人)
            return OutputRuleResult(
                action="dispatch", reason=detected_intent, text=text
            )

        # 如果距离 > 阈值，则不匹配
        return OutputRuleResult(action="allow", reason="No intent detected")

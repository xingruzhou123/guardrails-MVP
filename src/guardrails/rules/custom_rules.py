# src/guardrails/rules/custom_rules.py
import json
import re
import chromadb
from sentence_transformers import SentenceTransformer
from typing import Any, Dict, List, Optional

from .base import BaseOutputRule, OutputRuleResult
from guardrails.llms.base import BaseLLM

# The system prompt that instructs the LLM on how to behave as a classifier.
SYSTEM_PROMPT = """
You are an expert NLU model. Your task is to analyze the user's query and classify it into one of the predefined intents, and extract any relevant entities.

You MUST respond in a valid JSON format with two keys: "intent" and "entities".
- The "intent" value must be one of these exact strings: {intent_names}
- If the query does not match any intent, the "intent" value must be "none".
- The "entities" value must be a dictionary of extracted entities based on the intent's entity schema. If no entities are found, it should be an empty dictionary {{}}.
"""


class LLMClassifierRule(BaseOutputRule):
    """
    An input rule that uses a Language Model to classify user intent and extract entities.
    It implements a Retrieval-Augmented flow, similar to NeMo Guardrails, to provide
    the most relevant examples to the classification LLM.
    """

    def __init__(
        self,
        name: str,
        intents: List[Dict[str, Any]],
        shared_llm: BaseLLM,
        top_k: int = 5,
        **kwargs,
    ):
        """
        Initializes the rule.

        Args:
            name (str): The name of the rule.
            intents (List[Dict[str, Any]]): A list of intent definitions from the config.
            shared_llm (BaseLLM): The LLM client to use for classification.
            top_k (int): The number of relevant examples to retrieve for the prompt.
        """
        print("\n==============================================")
        print("  Initializing Retrieval-Augmented Guardrails ")
        print("==============================================")

        self.name = name
        self.classifier_llm = shared_llm
        self.top_k = top_k

        # --- THIS IS THE CRITICAL FIX: Handle old and new intent formats ---
        processed_intents = []
        for intent_item in intents:
            if isinstance(intent_item, str):
                # Handle old format (list of strings) by converting it
                programmatic_name = (
                    intent_item.lower().replace(" ", "_").replace("'", "")
                )
                processed_intents.append(
                    {
                        "name": programmatic_name,
                        "description": intent_item,
                        "examples": [
                            intent_item
                        ],  # Use the description as a single example
                    }
                )
            elif isinstance(intent_item, dict) and "name" in intent_item:
                # Handle new format (list of dicts)
                processed_intents.append(intent_item)

        self.intents = processed_intents
        self.intent_map = {i["name"]: i for i in self.intents}
        # --- END FIX ---

        # --- Vector Database Setup for Example Retrieval ---
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.client = chromadb.Client()

        collection_name = "intent_examples_collection_v4"
        try:
            self.client.delete_collection(name=collection_name)
        except Exception:
            pass  # Collection didn't exist, which is fine.

        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )
        self._initialize_vector_db()

    def _initialize_vector_db(self):
        """Encodes and indexes all examples from the config into ChromaDB."""
        all_examples = []
        all_metadatas = []
        all_ids = []
        example_count = 0

        for intent in self.intents:
            for example in intent.get("examples", []):
                all_examples.append(example)
                all_metadatas.append({"intent_name": intent["name"]})
                all_ids.append(f"ex_{example_count}")
                example_count += 1

        if not all_examples:
            print("[LLMClassifierRule] Warning: No examples found to index.")
            return

        print(
            f"[LLMClassifierRule] Encoding and indexing {len(all_examples)} examples in the vector database..."
        )
        embeddings = self.encoder.encode(all_examples)
        self.collection.add(
            embeddings=embeddings,
            documents=all_examples,
            metadatas=all_metadatas,
            ids=all_ids,
        )
        print("[LLMClassifierRule] Vector database is ready.")
        print(
            f"[LLMClassifierRule] Initialized with a retrieval-augmented flow for {len(self.intents)} intents."
        )

    def _format_examples_for_prompt(self, intents: List[Dict[str, Any]]) -> str:
        """Formats the retrieved intent definitions into a string for the LLM prompt."""
        formatted_string = "--- Relevant Intent Definitions ---\n"
        for intent in intents:
            formatted_string += f"\n- Intent Name: `{intent['name']}`\n"
            formatted_string += f"  - Description: {intent['description']}\n"
            if "entities" in intent:
                formatted_string += (
                    f"  - Entities Schema: {json.dumps(intent['entities'])}\n"
                )
            if "examples" in intent and intent["examples"]:
                formatted_string += "  - Examples:\n"
                for ex in intent["examples"]:
                    formatted_string += f'    - "{ex}"\n'
        formatted_string += "-----------------------------------\n"
        return formatted_string

    def _extract_json_from_response(self, response_text: str) -> Optional[str]:
        """
        Finds and extracts a JSON block from the LLM's potentially noisy response.
        """
        match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if match:
            return match.group(0)
        return None

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """
        Parses the LLM's string response into a dictionary.
        Now uses the robust JSON extraction method.
        """
        json_string = self._extract_json_from_response(response)

        if not json_string:
            return {"intent": "none", "entities": {}}

        try:
            return json.loads(json_string)
        except json.JSONDecodeError:
            return {"intent": "none", "entities": {}}

    async def apply(self, text: str, context: Dict[str, Any]) -> OutputRuleResult:
        """
        Applies the classification rule using the retrieval-augmented LLM flow.
        """
        # 1. Retrieve the most relevant examples from the vector database
        query_embedding = self.encoder.encode([text])
        results = self.collection.query(
            query_embeddings=query_embedding, n_results=self.top_k
        )

        retrieved_intent_names = set()
        if results and results["metadatas"]:
            for meta in results["metadatas"][0]:
                retrieved_intent_names.add(meta["intent_name"])

        # 2. Build the prompt with the retrieved examples
        relevant_intents = [
            self.intent_map[name]
            for name in retrieved_intent_names
            if name in self.intent_map
        ]

        if not relevant_intents:
            relevant_intents = self.intents

        examples_text = self._format_examples_for_prompt(relevant_intents)

        system_prompt = SYSTEM_PROMPT.format(
            intent_names=json.dumps(list(self.intent_map.keys()))
        )

        # --- Simplified and more direct user prompt ---
        user_prompt = f"""
{examples_text}
Based on the intent definitions above, analyze the following user query and respond with ONLY a valid JSON object.

User Query: "{text}"
"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # 3. Call the classifier LLM
        response_text = await self.classifier_llm.acomplete(messages=messages)

        # --- Add this line for crucial debugging ---
        print(f"[debug] Raw Classifier LLM Response:\n---\n{response_text}\n---")

        # 4. Parse the response
        classification = self._parse_llm_response(response_text)

        # 5. Return the result
        intent = classification.get("intent", "none")
        if intent != "none" and intent in self.intent_map:
            return OutputRuleResult(
                action="dispatch",
                reason="llm_intent_classifier",
                extra_info=classification,
            )

        return OutputRuleResult(
            action="allow", reason="llm_intent_classifier", extra_info=classification
        )

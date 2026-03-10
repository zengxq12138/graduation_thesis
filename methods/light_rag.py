"""
通过 HTTP API 调用 LightRAG。
"""
import sys
from pathlib import Path
from typing import List

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import Config
from .base import BaseMethod


class LightRAGMethod(BaseMethod):
    name = "light_rag"

    def __init__(self, config: Config = None):
        super().__init__(config)
        self.session = requests.Session()
        self._last_contexts: List[str] = []

    def _post_query(self, question: str, only_context: bool = False) -> str:
        payload = {
            "query": question,
            "mode": self.config.lightrag.mode,
        }
        if only_context:
            payload["only_need_context"] = True
            payload["top_k"] = self.config.lightrag.top_k
            payload["chunk_top_k"] = self.config.lightrag.chunk_top_k

        response = self.session.post(
            self.config.api.lightrag_url,
            json=payload,
            timeout=self.config.api.request_timeout,
        )
        response.raise_for_status()
        return response.json().get("response", "")

    def get_answer(self, question: str, max_chars: int = 200) -> str:
        answer = self._post_query(question, only_context=False)
        context = self._post_query(question, only_context=True)
        self._last_contexts = [context] if context else []
        return answer or "No answer found."

    def get_contexts(self, question: str) -> List[str]:
        return self._last_contexts

    def __del__(self):
        if hasattr(self, "session"):
            self.session.close()

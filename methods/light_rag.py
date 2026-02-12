"""
LightRAG 方法实现（通过 HTTP API 调用 LightRAG 服务）
"""
import sys
from pathlib import Path
from typing import List

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Config
from .base import BaseMethod


class LightRAGMethod(BaseMethod):
    """LightRAG 方法（通过 HTTP API）"""

    name = "light_rag"

    def __init__(self, config: Config = None):
        super().__init__(config)
        self.session = requests.Session()
        self._last_contexts = []  # 缓存最近一次检索的上下文

    def _query_lightrag(self, question: str, only_context: bool = False) -> str:
        """查询 LightRAG API"""
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
            json=payload
        )
        response.raise_for_status()
        result = response.json()
        return result.get("response", "")

    def get_answer(self, question: str, max_chars: int = 200) -> str:
        """获取问题的答案"""
        # 先获取答案
        answer = self._query_lightrag(question, only_context=False)

        # 同时获取上下文并缓存
        context_text = self._query_lightrag(question, only_context=True)
        self._last_contexts = [context_text] if context_text else []

        return answer if answer else "No answer found."

    def get_contexts(self, question: str) -> List[str]:
        """获取检索到的上下文"""
        # 返回缓存的上下文（已在 get_answer 中获取）
        return self._last_contexts

    def __del__(self):
        """关闭 session"""
        if hasattr(self, 'session'):
            self.session.close()

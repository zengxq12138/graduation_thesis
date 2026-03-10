"""
基于 Embedchain 的朴素 RAG。
"""
import os
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import Config
from .base import BaseMethod


class NaiveRAGMethod(BaseMethod):
    name = "naive_rag"

    def __init__(self, config: Config = None):
        super().__init__(config)
        self.app = None
        self._last_contexts: List[str] = []
        self._init_app()

    def _init_app(self) -> None:
        from embedchain import App

        if not self.config.api.openai_api_key:
            raise RuntimeError("请先设置环境变量 OPENAI_API_KEY")

        os.environ["OPENAI_API_KEY"] = self.config.api.openai_api_key
        os.environ["OPENAI_API_BASE"] = self.config.api.openai_base_url

        db_path = str(self.config.paths.db_dir)
        self.app = App.from_config(config=self.config.embedchain.to_dict(db_path))
        self._ensure_documents_loaded()

    def _ensure_documents_loaded(self) -> None:
        from embedchain import App

        document_path = self.config.get_document_path()
        db_count = self.app.db.count()
        if db_count > 0:
            print(f"向量数据库已有 {db_count} 条文档，跳过导入。")
            return

        if not document_path.exists():
            raise FileNotFoundError(f"知识库文档不存在: {document_path}")

        print(f"向量数据库为空，正在导入文档: {document_path}")
        self.app.reset()
        self.app = App.from_config(config=self.config.embedchain.to_dict(str(self.config.paths.db_dir)))
        self.app.add(str(document_path))

    def get_answer(self, question: str, max_chars: int = 200) -> str:
        answer = self.app.query(question)
        search_results = self.app.search(question, num_documents=self.config.embedchain.retrieve_top_k)
        self._last_contexts = [item["context"] for item in search_results if item.get("context")]
        return answer

    def get_contexts(self, question: str) -> List[str]:
        return self._last_contexts

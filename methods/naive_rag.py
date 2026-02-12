"""
Naive RAG 方法实现（基于 Embedchain）
"""
import os
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Config
from .base import BaseMethod


class NaiveRAGMethod(BaseMethod):
    """Naive RAG 方法（基于 Embedchain）"""

    name = "naive_rag"

    def __init__(self, config: Config = None):
        super().__init__(config)
        self.app = None
        self._last_contexts = []  # 缓存最近一次检索的上下文
        self._init_app()

    def _init_app(self):
        """初始化 Embedchain App"""
        from embedchain import App

        # 确保环境变量设置正确
        os.environ["OPENAI_API_KEY"] = self.config.api.openai_api_key
        os.environ["OPENAI_API_BASE"] = self.config.api.openai_base_url

        # 获取配置
        db_path = str(self.config.paths.db_dir)
        ec_config = self.config.embedchain.to_dict(db_path)

        print("正在初始化 Embedchain App 并加载向量数据库...")
        self.app = App.from_config(config=ec_config)

        # 检查向量数据库，若为空则导入文档
        self._ensure_documents_loaded()

    def _ensure_documents_loaded(self):
        """确保文档已加载到向量数据库"""
        from embedchain import App

        doc_path = self.config.get_document_path()
        db_count = self.app.db.count()

        if db_count == 0:
            print(f"向量数据库为空（{db_count} 条），正在重置并导入文档: {doc_path} ...")
            if doc_path.exists():
                self.app.reset()
                # reset 后需要重新初始化 App
                db_path = str(self.config.paths.db_dir)
                ec_config = self.config.embedchain.to_dict(db_path)
                self.app = App.from_config(config=ec_config)
                self.app.add(str(doc_path))
                new_count = self.app.db.count()
                print(f"文档导入完成！当前文档数: {new_count}")
                if new_count == 0:
                    print("警告: 导入后文档数仍为 0，请检查 OPENAI_API_KEY 和 embedding API 是否可用。")
            else:
                print(f"错误: 找不到文件 {doc_path}，请确保文件存在。")
        else:
            print(f"向量数据库已有 {db_count} 条文档，跳过导入。")

    def get_answer(self, question: str, max_chars: int = 200) -> str:
        """获取问题的答案"""
        answer = self.app.query(question)

        # 同时获取上下文并缓存
        search_results = self.app.search(question, num_documents=5)
        self._last_contexts = [r["context"] for r in search_results if "context" in r]

        return answer

    def get_contexts(self, question: str) -> List[str]:
        """获取检索到的上下文"""
        # 返回缓存的上下文（已在 get_answer 中获取）
        return self._last_contexts

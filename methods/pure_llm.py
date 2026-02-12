"""
Pure LLM 方法实现（不使用 RAG，直接调用大模型）
"""
import sys
from pathlib import Path
from typing import List

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Config
from .base import BaseMethod


class PureLLMMethod(BaseMethod):
    """纯 LLM 方法"""

    name = "pure_llm"

    def __init__(self, config: Config = None):
        super().__init__(config)
        self._init_client()

    def _init_client(self):
        """初始化 OpenAI 客户端"""
        api_key = self.config.api.openai_api_key
        if not api_key:
            raise RuntimeError("请设置环境变量 OPENAI_API_KEY")

        self.client = OpenAI(
            api_key=api_key,
            base_url=self.config.api.openai_base_url
        )

    def _build_prompt(self, question: str, max_chars: int) -> str:
        """构建提示词"""
        return (
            f"你是一个简洁的助手。请用中文回答以下问题，要求：回答不超过 {max_chars} 字（字数以中文字符计），"
            "不要超过限制；只返回答案内容，也不要包含标点前后的空行。\n\n"
            f"问题：{question}\n\n只返回答案："
        )

    def get_answer(self, question: str, max_chars: int = 200) -> str:
        """获取问题的答案"""
        prompt = self._build_prompt(question, max_chars)

        response = self.client.chat.completions.create(
            model=self.config.api.model_name,
            messages=[
                {"role": "system", "content": "你需要扮演一个果园病虫害的专家"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            stream=False,
        )
        return response.choices[0].message.content

    def get_contexts(self, question: str) -> List[str]:
        """Pure LLM 没有检索上下文，返回空列表"""
        return []

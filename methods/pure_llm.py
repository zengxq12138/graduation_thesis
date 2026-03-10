"""
纯大模型问答实现。
"""
import sys
from pathlib import Path
from typing import List

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import Config
from .base import BaseMethod


class PureLLMMethod(BaseMethod):
    name = "pure_llm"

    def __init__(self, config: Config = None):
        super().__init__(config)
        api_key = self.config.api.openai_api_key
        if not api_key:
            raise RuntimeError("请先设置环境变量 OPENAI_API_KEY")

        self.client = OpenAI(
            api_key=api_key,
            base_url=self.config.api.openai_base_url,
        )

    def build_prompt(self, question: str, max_chars: int) -> str:
        return (
            f"你是一个果园病虫害专家。请用中文回答以下问题，控制在 {max_chars} 字以内。"
            "只返回答案正文，不要输出额外说明或空行。\n\n"
            f"问题：{question}\n\n答案："
        )

    def get_answer(self, question: str, max_chars: int = 200) -> str:
        response = self.client.chat.completions.create(
            model=self.config.api.model_name,
            messages=[
                {"role": "system", "content": "你需要扮演一个果园病虫害的专家。"},
                {"role": "user", "content": self.build_prompt(question, max_chars)},
            ],
            temperature=0.0,
            stream=False,
        )
        return response.choices[0].message.content or ""

    def get_contexts(self, question: str) -> List[str]:
        return []

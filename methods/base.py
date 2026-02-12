"""
方法基类定义
"""
import json
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Config


@dataclass
class TestRecord:
    """测试记录"""
    question: str
    answer: str
    standard_answer: str
    contexts: List[str] = None

    def __post_init__(self):
        if self.contexts is None:
            self.contexts = []

    def to_dict(self) -> dict:
        return asdict(self)


class BaseMethod(ABC):
    """所有方法的基类"""

    name: str = "base"

    def __init__(self, config: Config = None):
        if config is None:
            from config import default_config
            config = default_config
        self.config = config

    @abstractmethod
    def get_answer(self, question: str, max_chars: int = 200) -> str:
        """获取问题的答案"""
        pass

    def get_contexts(self, question: str) -> List[str]:
        """获取检索到的上下文（默认返回空列表）"""
        return []

    def process_testset(self, test_type: str, verbose: bool = True) -> List[TestRecord]:
        """处理测试集"""
        testset_path = self.config.get_testset_path(test_type)
        output_path = self.config.get_output_path(self.name, test_type)
        max_chars = self.config.get_max_chars(test_type)

        if not testset_path.exists():
            raise FileNotFoundError(f"测试集文件不存在: {testset_path}")

        with open(testset_path, "r", encoding="utf-8") as f:
            test_data = json.load(f)

        records = []
        iterator = tqdm(test_data, desc=f"{self.name} - {test_type}") if verbose else test_data

        for item in iterator:
            question = item.get("问题", "")
            standard_answer = item.get("标准答案", "")

            if not question:
                continue

            try:
                answer = self.get_answer(question, max_chars)
                contexts = self.get_contexts(question)
                if verbose:
                    print(f"\nQ: {question}\nA: {answer}\n")
            except Exception as e:
                print(f"Error processing question: {question}\n{e}")
                answer = "Error occurred during processing."
                contexts = []

            record = TestRecord(
                question=question,
                answer=answer,
                standard_answer=standard_answer,
                contexts=contexts
            )
            records.append(record)

        # 保存结果
        self._save_results(records, output_path)
        return records

    def _save_results(self, records: List[TestRecord], output_path: Path):
        """保存测试结果"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = [r.to_dict() for r in records]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"结果已保存至: {output_path}")

    def run_all(self, verbose: bool = True) -> dict:
        """运行所有测试集"""
        results = {}
        for test_type in self.config.test_types:
            print(f"\n{'=' * 60}")
            print(f"正在运行 {self.name} - 测试集 {test_type}")
            print(f"{'=' * 60}")
            results[test_type] = self.process_testset(test_type, verbose)
        return results

"""
方法抽象层与测试集读写。
"""
import json
import sys
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, List

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import Config


QUESTION_KEYS = ("问题", "闂", "question")
STANDARD_ANSWER_KEYS = ("标准答案", "鏍囧噯绛旀", "standard_answer")


def _pick_value(item: dict[str, Any], candidates: tuple[str, ...]) -> str:
    for key in candidates:
        value = item.get(key)
        if value:
            return str(value)
    return ""


@dataclass
class TestRecord:
    question: str
    answer: str
    standard_answer: str
    contexts: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class BaseMethod(ABC):
    name: str = "base"

    def __init__(self, config: Config = None):
        if config is None:
            from config import default_config
            config = default_config
        self.config = config

    @abstractmethod
    def get_answer(self, question: str, max_chars: int) -> str:
        pass

    def get_contexts(self, question: str) -> List[str]:
        return []

    def load_testset(self, test_type: str) -> List[dict]:
        testset_path = self.config.get_testset_path(test_type)
        if not testset_path.exists():
            raise FileNotFoundError(f"测试集文件不存在: {testset_path}")

        with open(testset_path, "r", encoding="utf-8") as file:
            return json.load(file)

    def process_testset(self, test_type: str, verbose: bool = True) -> List[TestRecord]:
        output_path = self.config.get_output_path(self.name, test_type)
        test_data = self.load_testset(test_type)
        max_chars = self.config.get_max_chars(test_type)

        records: List[TestRecord] = []
        iterator = tqdm(test_data, desc=f"{self.name} - {test_type}") if verbose else test_data

        for item in iterator:
            question = _pick_value(item, QUESTION_KEYS)
            standard_answer = _pick_value(item, STANDARD_ANSWER_KEYS)
            if not question:
                continue

            try:
                answer = self.get_answer(question, max_chars=max_chars)
                contexts = self.get_contexts(question)
                if verbose:
                    print(f"\nQ: {question}\nA: {answer}\n")
            except Exception as exc:
                print(f"Error processing question: {question}\n{exc}")
                answer = "Error occurred during processing."
                contexts = []

            records.append(
                TestRecord(
                    question=question,
                    answer=answer,
                    standard_answer=standard_answer,
                    contexts=contexts,
                )
            )

        self.save_results(records, output_path)
        return records

    def save_results(self, records: List[TestRecord], output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump([record.to_dict() for record in records], file, ensure_ascii=False, indent=2)
        print(f"结果已保存至: {output_path}")

    def run_all(self, verbose: bool = True) -> dict[str, List[TestRecord]]:
        results = {}
        for test_type in self.config.test_types:
            print(f"\n{'=' * 60}")
            print(f"正在运行 {self.name} - 测试集 {test_type}")
            print(f"{'=' * 60}")
            results[test_type] = self.process_testset(test_type, verbose=verbose)
        return results

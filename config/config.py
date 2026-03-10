"""
统一配置与历史资产引导。
"""
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _copy_if_missing(source: Path, target: Path) -> None:
    if not source.exists() or target.exists():
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, target)
    else:
        shutil.copy2(source, target)


@dataclass
class APIConfig:
    openai_api_key: str = field(default_factory=lambda: os.getenv("QWEN_API_KEY", ""))
    openai_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model_name: str = "qwen3-max"
    judge_api_key: str = field(default_factory=lambda: os.getenv("ALI_CODING_PLAN", ""))
    judge_base_url: str = "https://coding.dashscope.aliyuncs.com/v1"
    judge_model_name: str = "glm-5"
    lightrag_url: str = "http://127.0.0.1:9621/query"
    request_timeout: int = 120


@dataclass
class PathConfig:
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    testset_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "testset")
    documents_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "documents")
    db_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "db")
    output_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "output")
    results_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "output" / "results")
    charts_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "output" / "charts")
    old_script_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "old_script")

    def ensure_dirs(self) -> None:
        for path in [
            self.data_dir,
            self.testset_dir,
            self.documents_dir,
            self.db_dir,
            self.output_dir,
            self.results_dir,
            self.charts_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    def bootstrap_from_legacy(self) -> None:
        self.ensure_dirs()

        for name in ("A.json", "B.json"):
            _copy_if_missing(
                self.old_script_dir / "pure_llm" / "testset" / name,
                self.testset_dir / name,
            )

        _copy_if_missing(
            self.old_script_dir / "naive_rag" / "经济果林病虫害防治手册.txt",
            self.documents_dir / "经济果林病虫害防治手册.txt",
        )

        legacy_db_dir = self.old_script_dir / "naive_rag" / "db"
        if legacy_db_dir.exists():
            for child in legacy_db_dir.iterdir():
                _copy_if_missing(child, self.db_dir / child.name)

        legacy_rating_dir = self.old_script_dir / "rating" / "output"
        if legacy_rating_dir.exists():
            for child in legacy_rating_dir.iterdir():
                if child.suffix.lower() == ".png":
                    _copy_if_missing(child, self.charts_dir / child.name)
                else:
                    _copy_if_missing(child, self.output_dir / child.name)

        legacy_result_dir = self.old_script_dir / "rating" / "testset"
        if legacy_result_dir.exists():
            for child in legacy_result_dir.glob("*_output_*.json"):
                _copy_if_missing(child, self.results_dir / child.name)


@dataclass
class EmbedchainConfig:
    llm_model: str = "qwen3-max"
    llm_temperature: float = 0.5
    llm_max_tokens: int = 1000
    embedder_model: str = "text-embedding-v4"
    vector_dimension: int = 1024
    chunk_size: int = 1000
    chunk_overlap: int = 100
    collection_name: str = "orchard-pest-rag"
    batch_size: int = 10
    retrieve_top_k: int = 5

    def to_dict(self, db_path: str) -> dict:
        return {
            "llm": {
                "provider": "openai",
                "config": {
                    "model": self.llm_model,
                    "temperature": self.llm_temperature,
                    "max_tokens": self.llm_max_tokens,
                    "top_p": 1,
                    "stream": False,
                },
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": self.embedder_model,
                    "vector_dimension": self.vector_dimension,
                },
            },
            "chunker": {
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "length_function": "len",
            },
            "vectordb": {
                "provider": "chroma",
                "config": {
                    "collection_name": self.collection_name,
                    "dir": db_path,
                    "batch_size": self.batch_size,
                },
            },
        }


@dataclass
class LightRAGConfig:
    mode: str = "mix"
    top_k: int = 5
    chunk_top_k: int = 5


@dataclass
class Config:
    api: APIConfig = field(default_factory=APIConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    embedchain: EmbedchainConfig = field(default_factory=EmbedchainConfig)
    lightrag: LightRAGConfig = field(default_factory=LightRAGConfig)
    test_types: List[str] = field(default_factory=lambda: ["A", "B"])
    methods: List[str] = field(default_factory=lambda: ["pure_llm", "naive_rag", "light_rag"])
    max_answer_chars_A: int = 100
    max_answer_chars_B: int = 350

    def __post_init__(self) -> None:
        self.paths.bootstrap_from_legacy()
        if self.api.openai_api_key:
            os.environ["OPENAI_API_KEY"] = self.api.openai_api_key
        os.environ["OPENAI_API_BASE"] = self.api.openai_base_url

    def get_testset_path(self, test_type: str) -> Path:
        return self.paths.testset_dir / f"{test_type}.json"

    def get_output_path(self, method: str, test_type: str) -> Path:
        return self.paths.results_dir / f"{method}_output_{test_type}.json"

    def get_document_path(self) -> Path:
        return self.paths.documents_dir / "经济果林病虫害防治手册.txt"

    def get_max_chars(self, test_type: str) -> int:
        return self.max_answer_chars_A if test_type == "A" else self.max_answer_chars_B


default_config = Config()

"""
统一配置模块
"""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent


@dataclass
class APIConfig:
    """API 配置"""
    # OpenAI 兼容 API 配置（用于 RAG 生成）
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model_name: str = "qwen3-max"

    # 评分模型配置（使用 GLM-4.7）
    # judge_api_key: str = field(default_factory=lambda: os.getenv("ZAI_API_KEY") or os.getenv("OPENAI_API_KEY", ""))
    # judge_base_url: str = "https://open.bigmodel.cn/api/paas/v4/"
    # judge_model_name: str = "glm-4.7"

    judge_api_key: str = field(default_factory=lambda: os.getenv("DMX"))
    judge_base_url: str = "https://www.dmxapi.com/v1"
    judge_model_name: str = "glm-4.7"

    # LightRAG 服务配置
    lightrag_url: str = "http://127.0.0.1:9621/query"


@dataclass
class PathConfig:
    """路径配置"""
    # 数据目录
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    testset_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "testset")
    documents_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "documents")

    # 输出目录
    output_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "output")
    results_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "output" / "results")
    charts_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "output" / "charts")

    # 数据库目录（Naive RAG）
    db_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "db")

    def ensure_dirs(self):
        """确保所有目录存在"""
        for path in [self.data_dir, self.testset_dir, self.documents_dir,
                     self.output_dir, self.results_dir, self.charts_dir, self.db_dir]:
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class EmbedchainConfig:
    """Embedchain (Naive RAG) 配置"""
    llm_model: str = "qwen3-max"
    llm_temperature: float = 0.5
    llm_max_tokens: int = 1000

    embedder_model: str = "text-embedding-v4"
    vector_dimension: int = 1024

    chunk_size: int = 1000
    chunk_overlap: int = 100

    collection_name: str = "orchard-pest-rag"
    batch_size: int = 10  # DashScope 限制

    def to_dict(self, db_path: str) -> dict:
        """转换为 Embedchain 配置字典"""
        return {
            'llm': {
                'provider': 'openai',
                'config': {
                    'model': self.llm_model,
                    'temperature': self.llm_temperature,
                    'max_tokens': self.llm_max_tokens,
                    'top_p': 1,
                    'stream': False,
                }
            },
            'embedder': {
                'provider': 'openai',
                'config': {
                    'model': self.embedder_model,
                    'vector_dimension': self.vector_dimension,
                }
            },
            'chunker': {
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap,
                'length_function': 'len'
            },
            'vectordb': {
                'provider': 'chroma',
                'config': {
                    'collection_name': self.collection_name,
                    'dir': db_path,
                    'batch_size': self.batch_size,
                }
            }
        }


@dataclass
class LightRAGConfig:
    """LightRAG 配置"""
    mode: str = "mix"  # mix/local/global
    top_k: int = 5
    chunk_top_k: int = 5


@dataclass
class Config:
    """主配置类"""
    api: APIConfig = field(default_factory=APIConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    embedchain: EmbedchainConfig = field(default_factory=EmbedchainConfig)
    lightrag: LightRAGConfig = field(default_factory=LightRAGConfig)

    # 测试集配置
    test_types: List[str] = field(default_factory=lambda: ["A", "B"])

    # 方法配置
    methods: List[str] = field(default_factory=lambda: ["pure_llm", "naive_rag", "light_rag"])

    # 答案长度限制
    max_answer_chars_A: int = 100  # 数据集 A 的回答长度限制
    max_answer_chars_B: int = 350  # 数据集 B 的回答长度限制

    def __post_init__(self):
        """初始化后设置环境变量"""
        if self.api.openai_api_key:
            os.environ["OPENAI_API_KEY"] = self.api.openai_api_key
        os.environ["OPENAI_API_BASE"] = self.api.openai_base_url
        self.paths.ensure_dirs()

    def get_testset_path(self, test_type: str) -> Path:
        """获取测试集路径"""
        return self.paths.testset_dir / f"{test_type}.json"

    def get_output_path(self, method: str, test_type: str) -> Path:
        """获取输出文件路径"""
        return self.paths.results_dir / f"{method}_output_{test_type}.json"

    def get_document_path(self) -> Path:
        """获取知识库文档路径"""
        return self.paths.documents_dir / "经济果林病虫害防治手册.txt"

    def get_max_chars(self, test_type: str) -> int:
        """根据测试集类型获取答案长度限制"""
        return self.max_answer_chars_A if test_type == "A" else self.max_answer_chars_B


# 全局默认配置实例
default_config = Config()

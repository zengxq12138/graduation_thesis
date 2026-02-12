from .base import BaseMethod, TestRecord
from .light_rag import LightRAGMethod
from .naive_rag import NaiveRAGMethod
from .pure_llm import PureLLMMethod

# 方法注册表
METHOD_REGISTRY = {
    "pure_llm": PureLLMMethod,
    "naive_rag": NaiveRAGMethod,
    "light_rag": LightRAGMethod,
}


def get_method(method_name: str, config=None):
    """获取指定方法的实例"""
    if method_name not in METHOD_REGISTRY:
        raise ValueError(f"未知的方法: {method_name}。可用方法: {list(METHOD_REGISTRY.keys())}")
    return METHOD_REGISTRY[method_name](config)

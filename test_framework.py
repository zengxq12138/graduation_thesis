#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
项目功能自检脚本。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config


def print_section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(title)
    print(f"{'=' * 60}")


def test_config():
    print_section("1. 测试配置模块")
    config = Config()
    print(f"API Base URL: {config.api.openai_base_url}")
    print(f"Model: {config.api.model_name}")
    print(f"Judge API: {config.api.judge_base_url}")
    print(f"Judge Model: {config.api.judge_model_name}")
    print(f"测试集目录: {config.paths.testset_dir}")
    print(f"输出目录: {config.paths.output_dir}")
    print(f"测试集 A 存在: {config.get_testset_path('A').exists()}")
    print(f"测试集 B 存在: {config.get_testset_path('B').exists()}")
    print(f"知识库文档存在: {config.get_document_path().exists()}")
    print(f"环境变量 OPENAI_API_KEY: {'已设置' if config.api.openai_api_key else '未设置'}")
    print(f"环境变量 DMX: {'已设置' if config.api.judge_api_key else '未设置'}")
    return config


def test_methods():
    print_section("2. 测试方法模块")
    from methods import METHOD_REGISTRY
    print(f"可用方法: {list(METHOD_REGISTRY.keys())}")
    return True


def test_evaluation():
    print_section("3. 测试评估模块")
    from evaluation import Evaluator, plot_results
    print("评估模块导入成功")
    print("绘图模块导入成功")
    return True


def main():
    print_section("RAG 评测框架功能测试")
    test_config()
    test_methods()
    test_evaluation()
    print_section("测试完成")
    print("提示: 运行需要设置环境变量 OPENAI_API_KEY 和 DMX")


if __name__ == "__main__":
    main()

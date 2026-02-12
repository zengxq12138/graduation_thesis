#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
功能测试脚本
测试重构后的 RAG 评测框架各个组件是否正常工作
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from config import Config


def print_section(title: str):
    """打印分隔标题"""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def test_config():
    """测试配置模块"""
    print_section("1. 测试配置模块")

    try:
        config = Config()
        print(f"✓ 配置初始化成功")
        print(f"  - API Base URL: {config.api.openai_base_url}")
        print(f"  - Model: {config.api.model_name}")
        print(f"  - 测试集目录: {config.paths.testset_dir}")
        print(f"  - 输出目录: {config.paths.output_dir}")
        print(f"  - 测试集 A 路径存在: {config.get_testset_path('A').exists()}")
        print(f"  - 知识库文档存在: {config.get_document_path().exists()}")
        return config
    except Exception as e:
        print(f"✗ 配置初始化失败: {e}")
        return None


def test_pure_llm(config: Config, question: str):
    """测试 Pure LLM 方法"""
    print_section("2. 测试 Pure LLM 方法")

    try:
        from methods import get_method
        method = get_method("pure_llm", config)
        print(f"✓ Pure LLM 方法初始化成功")

        print(f"\n测试问题: {question}")
        answer = method.get_answer(question, max_chars=100)
        contexts = method.get_contexts(question)

        print(f"回答: {answer}")
        print(f"上下文数量: {len(contexts)} (Pure LLM 应该为 0)")

        return {"question": question, "answer": answer, "standard_answer": "测试标准答案", "contexts": contexts}
    except Exception as e:
        print(f"✗ Pure LLM 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_naive_rag(config: Config, question: str):
    """测试 Naive RAG 方法"""
    print_section("3. 测试 Naive RAG 方法")

    try:
        from methods import get_method
        method = get_method("naive_rag", config)
        print(f"✓ Naive RAG 方法初始化成功")

        print(f"\n测试问题: {question}")
        answer = method.get_answer(question, max_chars=200)
        contexts = method.get_contexts(question)

        print(f"回答: {answer[:200]}..." if len(answer) > 200 else f"回答: {answer}")
        print(f"上下文数量: {len(contexts)}")
        if contexts:
            print(f"上下文片段预览: {contexts[0][:100]}...")

        return {"question": question, "answer": answer, "standard_answer": "测试标准答案", "contexts": contexts}
    except Exception as e:
        print(f"✗ Naive RAG 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_light_rag(config: Config, question: str):
    """测试 LightRAG 方法"""
    print_section("4. 测试 LightRAG 方法")

    print("注意: LightRAG 需要后台服务运行在 http://127.0.0.1:9621")

    try:
        from methods import get_method
        method = get_method("light_rag", config)
        print(f"✓ LightRAG 方法初始化成功")

        print(f"\n测试问题: {question}")
        answer = method.get_answer(question, max_chars=200)
        contexts = method.get_contexts(question)

        print(f"回答: {answer[:200]}..." if len(answer) > 200 else f"回答: {answer}")
        print(f"上下文数量: {len(contexts)}")
        if contexts and contexts[0]:
            print(f"上下文片段预览: {contexts[0][:100]}...")

        return {"question": question, "answer": answer, "standard_answer": "测试标准答案", "contexts": contexts}
    except Exception as e:
        print(f"✗ LightRAG 测试失败: {e}")
        print("  (如果是连接错误，请确保 LightRAG 服务已启动)")
        import traceback
        traceback.print_exc()
        return None


def test_evaluator(config: Config, test_entry: dict):
    """测试评分模块"""
    print_section("5. 测试 LLM Judge 评分")

    if test_entry is None:
        print("✗ 没有可用的测试数据，跳过评分测试")
        return None

    try:
        from evaluation.evaluator import Evaluator
        evaluator = Evaluator(config)
        print(f"✓ Evaluator 初始化成功")
        print(f"  - 评分模型: {config.api.judge_model_name}")
        print(f"  - API URL: {config.api.judge_base_url}")

        print(f"\n正在评分...")
        print(f"  问题: {test_entry['question']}")
        print(f"  回答: {test_entry['answer'][:100]}...")

        result = evaluator._evaluate_single(test_entry)

        print(f"\n评分结果:")
        print(f"  - 忠实度 (Faithfulness): {result.get('faithfulness_score', 'N/A')}")
        print(f"  - 完整性 (Comprehensiveness): {result.get('comprehensiveness_score', 'N/A')}")
        print(f"  - 相关性 (Relevance): {result.get('relevance_score', 'N/A')}")
        print(f"  - 评语: {result.get('reason', 'N/A')}")

        return result
    except Exception as e:
        print(f"✗ 评分测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_plotter(config: Config):
    """测试绘图模块"""
    print_section("6. 测试绘图模块")

    try:
        from evaluation.plotter import load_results, setup_plot_style
        import pandas as pd

        # 加载现有数据
        progress_file = config.paths.output_dir / "evaluation_progress.jsonl"
        results = load_results(progress_file)

        if not results:
            print("✗ 没有找到评测数据文件，跳过绘图测试")
            return False

        print(f"✓ 成功加载 {len(results)} 条评测记录")

        # 测试绘图样式设置
        setup_plot_style()
        print("✓ 绘图样式设置成功")

        # 测试 DataFrame 转换
        df = pd.DataFrame(results)
        print(f"✓ DataFrame 创建成功，形状: {df.shape}")

        # 执行完整绘图
        from evaluation import plot_results
        plot_results(config)

        # 验证图表文件
        charts_dir = config.paths.charts_dir
        expected_charts = [
            "chart_comprehensiveness.png",
            "chart_faithfulness.png",
            "chart_relevance.png",
            "chart_heatmap.png"
        ]

        all_exist = True
        for chart in expected_charts:
            chart_path = charts_dir / chart
            if chart_path.exists():
                print(f"✓ {chart} 已生成")
            else:
                print(f"✗ {chart} 未找到")
                all_exist = False

        return all_exist
    except Exception as e:
        print(f"✗ 绘图测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试流程"""
    print("\n" + "#" * 60)
    print("#  RAG 评测框架功能测试")
    print("#" * 60)

    # 测试问题
    test_question = "柑橘疮痂病的病原菌是什么？"

    # 1. 测试配置
    config = test_config()
    if config is None:
        print("\n配置测试失败，无法继续")
        return

    # 收集测试结果用于评分
    test_entries = []

    # 2. 测试 Pure LLM
    entry = test_pure_llm(config, test_question)
    if entry:
        test_entries.append(("pure_llm", entry))

    # 3. 测试 Naive RAG
    entry = test_naive_rag(config, test_question)
    if entry:
        test_entries.append(("naive_rag", entry))

    # 4. 测试 LightRAG
    entry = test_light_rag(config, test_question)
    if entry:
        test_entries.append(("light_rag", entry))

    # 5. 测试评分（使用第一个成功的结果）
    if test_entries:
        method_name, test_entry = test_entries[0]
        # 添加真实的标准答案
        test_entry["standard_answer"] = "柑橘疮痂病的病原菌是柑橘痂囊腔菌（Elsinoe fawcettii），其无性阶段为柑橘痂圆孢菌。"
        print(f"\n使用 {method_name} 的结果进行评分测试")
        test_evaluator(config, test_entry)
    else:
        print("\n没有成功的方法测试结果，跳过评分测试")

    # 6. 测试绘图
    test_plotter(config)

    # 总结
    print_section("测试完成")
    print("请检查上述各项测试结果。")
    print("如有 ✗ 标记的项目，请检查相关配置或依赖。")


if __name__ == "__main__":
    main()

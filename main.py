#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAG 评测框架主入口

支持的命令：
    python main.py run --method pure_llm --testset A,B    # 运行指定方法
    python main.py run --all                              # 运行所有方法
    python main.py evaluate                               # 评估所有结果
    python main.py plot                                   # 绘制图表
    python main.py pipeline                               # 完整流程（运行+评估+绘图）
"""
import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from methods import get_method, METHOD_REGISTRY
from evaluation import Evaluator, plot_results


def cmd_run(args, config: Config):
    """运行指定方法生成回答"""
    if args.all:
        methods_to_run = config.methods
    elif args.method:
        methods_to_run = [m.strip() for m in args.method.split(",")]
    else:
        print("请指定要运行的方法 (--method) 或使用 --all 运行所有方法")
        return

    test_types = [t.strip() for t in args.testset.split(",")] if args.testset else config.test_types

    # 临时修改配置
    config.test_types = test_types

    for method_name in methods_to_run:
        if method_name not in METHOD_REGISTRY:
            print(f"警告: 未知方法 '{method_name}'，跳过。可用方法: {list(METHOD_REGISTRY.keys())}")
            continue

        print(f"\n{'#' * 70}")
        print(f"# 正在运行方法: {method_name}")
        print(f"{'#' * 70}")

        try:
            method = get_method(method_name, config)
            method.run_all(verbose=not args.quiet)
        except Exception as e:
            print(f"运行 {method_name} 时出错: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()


def cmd_evaluate(args, config: Config):
    """评估所有方法的结果"""
    print("\n" + "=" * 70)
    print("开始评估流程")
    print("=" * 70)

    evaluator = Evaluator(config)
    results = evaluator.evaluate_all()

    if results:
        summary = evaluator.get_summary(results)
        print("\n评测完成！")


def cmd_plot(args, config: Config):
    """绘制评测结果图表"""
    print("\n" + "=" * 70)
    print("开始绘制图表")
    print("=" * 70)

    plot_results(config)


def cmd_pipeline(args, config: Config):
    """完整流程：运行 + 评估 + 绘图"""
    print("\n" + "#" * 70)
    print("# 开始完整评测流程")
    print("#" * 70)

    # 1. 运行所有方法
    if not args.skip_run:
        args.all = True
        args.testset = None
        cmd_run(args, config)

    # 2. 评估
    if not args.skip_eval:
        cmd_evaluate(args, config)

    # 3. 绘图
    if not args.skip_plot:
        cmd_plot(args, config)

    print("\n" + "#" * 70)
    print("# 完整流程执行完毕！")
    print("#" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="RAG 评测框架",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py run --method pure_llm --testset A      # 只运行 Pure LLM 方法，测试集 A
  python main.py run --method naive_rag,light_rag      # 运行多个方法
  python main.py run --all                             # 运行所有方法
  python main.py evaluate                              # 评估已生成的结果
  python main.py plot                                  # 绘制图表
  python main.py pipeline                              # 完整流程
  python main.py pipeline --skip-run                   # 跳过生成，只评估和绘图
        """
    )

    parser.add_argument("--debug", action="store_true", help="显示详细错误信息")

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # run 命令
    run_parser = subparsers.add_parser("run", help="运行方法生成回答")
    run_parser.add_argument("--method", "-m", type=str, help="要运行的方法，多个用逗号分隔")
    run_parser.add_argument("--all", "-a", action="store_true", help="运行所有方法")
    run_parser.add_argument("--testset", "-t", type=str, help="测试集类型，如 A,B")
    run_parser.add_argument("--quiet", "-q", action="store_true", help="静默模式，不打印详细输出")

    # evaluate 命令
    eval_parser = subparsers.add_parser("evaluate", help="评估结果")

    # plot 命令
    plot_parser = subparsers.add_parser("plot", help="绘制图表")

    # pipeline 命令
    pipe_parser = subparsers.add_parser("pipeline", help="完整流程")
    pipe_parser.add_argument("--skip-run", action="store_true", help="跳过运行步骤")
    pipe_parser.add_argument("--skip-eval", action="store_true", help="跳过评估步骤")
    pipe_parser.add_argument("--skip-plot", action="store_true", help="跳过绘图步骤")
    pipe_parser.add_argument("--quiet", "-q", action="store_true", help="静默模式")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # 初始化配置
    try:
        config = Config()
    except Exception as e:
        print(f"配置初始化失败: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return

    # 执行对应命令
    commands = {
        "run": cmd_run,
        "evaluate": cmd_evaluate,
        "plot": cmd_plot,
        "pipeline": cmd_pipeline,
    }

    try:
        commands[args.command](args, config)
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"\n程序执行出错: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()

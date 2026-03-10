#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAG 评测框架主入口。
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config
from evaluation import Evaluator, plot_results
from methods import METHOD_REGISTRY, get_method


def parse_csv_argument(raw_value: str | None, fallback: list[str]) -> list[str]:
    if not raw_value:
        return list(fallback)
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def cmd_run(args, config: Config) -> None:
    methods_to_run = config.methods if args.all else parse_csv_argument(args.method, [])
    if not methods_to_run:
        raise ValueError("请指定 --method 或使用 --all")

    config.test_types = parse_csv_argument(args.testset, config.test_types)
    for method_name in methods_to_run:
        if method_name not in METHOD_REGISTRY:
            print(f"警告: 未知方法 '{method_name}'，跳过。可用方法: {list(METHOD_REGISTRY.keys())}")
            continue
        method = get_method(method_name, config)
        method.run_all(verbose=not args.quiet)


def cmd_evaluate(args, config: Config) -> None:
    evaluator = Evaluator(config)
    # 评估并生成原始 CSV（不包含修正）
    evaluator.evaluate_all(apply_fix=False)


def cmd_fix(args, config: Config) -> None:
    """对评估结果进行数据修正"""
    from evaluation.evaluator import apply_fix_to_csv
    apply_fix_to_csv(config)


def cmd_plot(args, config: Config) -> None:
    plot_results(config)


def cmd_pipeline(args, config: Config) -> None:
    if not args.skip_run:
        cmd_run(argparse.Namespace(all=True, method=None, testset=None, quiet=args.quiet), config)
    if not args.skip_eval:
        cmd_evaluate(args, config)
    if not args.skip_fix:
        cmd_fix(args, config)
    if not args.skip_plot:
        cmd_plot(args, config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAG 评测框架")
    parser.add_argument("--debug", action="store_true", help="显示详细错误信息")

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    run_parser = subparsers.add_parser("run", help="运行方法生成回答")
    run_parser.add_argument("--method", "-m", type=str, help="要运行的方法，多个用逗号分隔")
    run_parser.add_argument("--all", "-a", action="store_true", help="运行所有方法")
    run_parser.add_argument("--testset", "-t", type=str, help="测试集类型，如 A,B")
    run_parser.add_argument("--quiet", "-q", action="store_true", help="静默模式")

    subparsers.add_parser("evaluate", help="评估结果")
    subparsers.add_parser("fix", help="应用数据修正规则")
    subparsers.add_parser("plot", help="绘制图表")

    pipeline_parser = subparsers.add_parser("pipeline", help="完整流程")
    pipeline_parser.add_argument("--skip-run", action="store_true", help="跳过运行步骤")
    pipeline_parser.add_argument("--skip-eval", action="store_true", help="跳过评估步骤")
    pipeline_parser.add_argument("--skip-fix", action="store_true", help="跳过修正步骤")
    pipeline_parser.add_argument("--skip-plot", action="store_true", help="跳过绘图步骤")
    pipeline_parser.add_argument("--quiet", "-q", action="store_true", help="静默模式")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    config = Config()
    commands = {
        "run": cmd_run,
        "evaluate": cmd_evaluate,
        "fix": cmd_fix,
        "plot": cmd_plot,
        "pipeline": cmd_pipeline,
    }

    try:
        commands[args.command](args, config)
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as exc:
        print(f"\n程序执行出错: {exc}")
        if args.debug:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
提取LightRAG文档的关键内容
"""
from pathlib import Path
import re
import sys

# 设置输出编码
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def extract_key_sections():
    """提取LightRAG文档的关键章节"""

    # 读取文本文件
    txt_file = Path('thesis/参考文档/lighRAG.txt')
    text = txt_file.read_text(encoding='utf-8')

    # 定义关键词
    keywords = [
        'Abstract',
        'Introduction',
        'Preliminary',
        'Methodology',
        'Dual-Level',
        'Graph-Enhanced',
        'Lightweight',
        'Retrieval',
        'Indexing'
    ]

    print("="*80)
    print("LightRAG文档关键内容提取")
    print("="*80)

    # 提取摘要部分
    print("\n1. 摘要 (Abstract)")
    print("-"*80)
    abstract_start = text.find('Abstract')
    if abstract_start != -1:
        abstract_end = text.find('Introduction', abstract_start)
        if abstract_end == -1:
            abstract_end = abstract_start + 2000
        abstract = text[abstract_start:abstract_end]
        # 清理文本
        abstract = re.sub(r'\s+', ' ', abstract)
        print(abstract[:1500])

    # 提取介绍部分
    print("\n\n2. 介绍 (Introduction)")
    print("-"*80)
    intro_start = text.find('Introduction')
    if intro_start != -1:
        intro_end = text.find('Preliminary', intro_start)
        if intro_end == -1:
            intro_end = intro_start + 3000
        intro = text[intro_start:intro_end]
        intro = re.sub(r'\s+', ' ', intro)
        print(intro[:2000])

    # 搜索Dual-Level相关内容
    print("\n\n3. 双层检索机制 (Dual-Level Retrieval)")
    print("-"*80)
    dual_level_matches = []
    for match in re.finditer(r'.{200}dual.level.{500}', text, re.IGNORECASE):
        dual_level_matches.append(match.group())

    for i, match in enumerate(dual_level_matches[:3]):
        print(f"\n匹配 {i+1}:")
        cleaned = re.sub(r'\s+', ' ', match)
        print(cleaned)

    # 搜索Graph-Enhanced相关内容
    print("\n\n4. 图增强索引 (Graph-Enhanced Indexing)")
    print("-"*80)
    graph_matches = []
    for match in re.finditer(r'.{200}graph.enhanced.{500}', text, re.IGNORECASE):
        graph_matches.append(match.group())

    for i, match in enumerate(graph_matches[:3]):
        print(f"\n匹配 {i+1}:")
        cleaned = re.sub(r'\s+', ' ', match)
        print(cleaned)

    # 搜索公式和算法相关内容
    print("\n\n5. 核心算法与公式")
    print("-"*80)

    # 查找包含equation, formula, algorithm的内容
    algo_matches = []
    for match in re.finditer(r'.{150}(equation|formula|algorithm|Equation|Formula|Algorithm).{300}', text):
        algo_matches.append(match.group())

    for i, match in enumerate(algo_matches[:5]):
        print(f"\n匹配 {i+1}:")
        cleaned = re.sub(r'\s+', ' ', match)
        print(cleaned)

if __name__ == "__main__":
    extract_key_sections()
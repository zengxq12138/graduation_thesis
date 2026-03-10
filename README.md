# RAG 评测框架

这是一个用于比较不同 RAG（检索增强生成）方法的评测框架，专门针对果园病虫害 Cultural领域I'll analyze the current project state and continue refactoring it to make it runnable. Let me focus on creating a simple README and checking if there are any other issues.<tool_call>Write<arg_key>content</arg_key><arg_value># RAG 评测框架

这是一个用于比较不同 RAG（检索增强生成）方法的评测框架，专门针对果园病虫害防治领域的问答系统。

## 项目概述

本框架比较了三种不同的问答方法：

1. **Pure LLM**: 纯大模型问答，不使用检索
2. **Naive RAG**: 基于 Embedchain 的朴素 RAG 实现
3. **Light RAG**: 使用外部 LightRAG 服务的图 RAG 实现

## 环境要求

- Python 3.12+
- 必需的环境变量：
  - `OPENAI_API_KEY`: 用于生成回答（DashScope/Qwen）
  - `DMX`: 用于 LLM Judge 评分（DMX API with GLM-4.7）
- LightRAG 服务（可选，用于 light_rag 方法）：
  - 默认地址：`http://127.0.0.1:9621/query`

## 安装依赖

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 功能测试

```bash
python test_framework.py
```

这将检查：
- 配置文件是否正确
- 测试集文件是否存在
- 环境变量是否设置
- 各模块是否可以正常导入

### 2. 运行方法生成回答

```bash
# 运行单个方法
python main.py run --method pure_llm --testset A

# 运行多个方法
python main.py run --method pure_llm,naive_rag --testset A,B

# 运行所有方法
python main.py run --all
```

### 3. 运行评估

```bash
python main.py evaluate
```

这将使用 LLM Judge 对所有已生成的回答进行评分。

### 4. 生成图表

```bash
python main.py plot
```

这将生成评分对比图表和热力图。

### 5. 完整流程

```bash
python main.py pipeline
```

这将依次执行：运行方法 → 评估 → 绘图。

## 项目结构

```
config/          → 配置模块（API keys, paths, model settings）
methods/         → 方法实现（BaseMethod, PureLLM, NaiveRAG, LightRAG）
evaluation/      → 评估模块（LLM Judge, 绘图）
data/            → 测试集和知识库文档
  testset/       → 测试集（A.json, B.json）
  documents/     → 知识库文档
  db/            → 向量数据库
output/          → 输出结果
  results/       → 各方法的回答结果
  charts/        → 生成的图表
  evaluation_progress.jsonl  → 评估进度
  final_evaluation_results.csv → 最终评估结果
```

## 评分维度

LLM Judge 从三个维度对回答进行评分（1-10分）：

1. **忠实度 (Faithfulness)**: 回答是否严格基于参考资料，是否存在幻觉
2. **完整性 (Comprehensiveness)**: 是否涵盖了标准答案的所有关键要点
3. **有用性 (Relevance)**: 回答是否直接解决了问题，没有废话

## 常见问题

### Q: 运行时提示"请先设置环境变量 OPENAI_API_KEY"

A: 需要设置环境变量：
```bash
export OPENAI_API_KEY="your-api-key"
export DMX="your-dmx-api-key"
```

### Q: LightRAG 方法无法连接

A: 确保 LightRAG 服务正在运行，默认地址为 `http://127.0.0.1:9621/query`。可以通过修改 `config/config.py` 中的 `lightrag_url` 来更改地址。

### Q: 评估过程中断怎么办？

A: 评估模块支持断点续传，会自动跳过已评估的问题。可以直接重新运行 `python main.py evaluate`。

## 扩展方法

要添加新的方法，需要：

1. 在 `methods/` 目录下创建新文件，继承 `BaseMethod` 类
2. 实现 `get_answer(question: str, max_chars: int) -> str` 方法
3. 可选实现 `get_contexts(question: str) -> List[str]` 方法
4. 在 `methods/__init__.py` 中注册到 `METHOD_REGISTRY`

## 许可证

本项目仅供学术研究使用。
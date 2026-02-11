import os
import uvicorn
from embedchain import App
from fastapi import FastAPI
from pydantic import BaseModel
import sys
import traceback
import json

# 1. 安全配置：建议从环境变量读取，不要硬编码
# 这里的 Key 仅作示例，实际运行请替换为你重新生成的新 Key
os.environ["OPENAI_API_KEY"] =os.getenv("OPENAI_API_KEY")
# 使用环境变量指定 OpenAI-compatible API base（优先级：已有环境变量 > 默认 dashscope URL）
os.environ["OPENAI_API_BASE"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 2. 定义配置
config = {
    # --- LLM 配置 ---
    'llm': {
        'provider': 'openai',
        'config': {
            'model': 'qwen3-max', 
            'temperature': 0.5,
            'max_tokens': 1000,
            'top_p': 1,
            'stream': False,
            # If you need a custom OpenAI-compatible base URL, set it via
            # the environment variable `OPENAI_API_BASE` instead of using
            # `base_url` here to match the installed embedchain version.
        }
    },
    # --- Embedder 配置 ---
    'embedder': {
        'provider': 'openai',
        'config': {
            'model': 'text-embedding-v4',
            'vector_dimension': 1024,
            # For embedder API base, use `OPENAI_API_BASE` env var if needed.
        }
    },
    'chunker': {
        'chunk_size': 1000,
        'chunk_overlap': 100,
        'length_function': 'len'
    },
    'vectordb': {
        'provider': 'chroma',
        'config': {
            'collection_name': 'orchard-pest-rag',
            'dir': 'db',
            'batch_size': 10,  # DashScope text-embedding-v4 限制每批最多 10 条
        }
    }
}

# 3. 初始化 RAG 应用
# 注意：第一次运行时会自动下载/创建数据库，速度取决于文档大小
app = App.from_config(config=config)


# 4. 自动检测向量数据库，若为空则导入文档
txt_path = "./经济果林病虫害防治手册.txt"
db_count = app.db.count()
if db_count == 0:
    print(f"向量数据库为空（{db_count} 条），正在重置并导入文档: {txt_path} ...")
    if os.path.exists(txt_path):
        app.reset()
        app = App.from_config(config=config)
        app.add(txt_path)
        new_count = app.db.count()
        print(f"文档导入完成！当前文档数: {new_count}")
        if new_count == 0:
            print("警告: 导入后文档数仍为 0，请检查 OPENAI_API_KEY 和 embedding API 是否可用。")
    else:
        print(f"错误: 找不到文件 {txt_path}，请确保文件在当前目录下。")
else:
    print(f"向量数据库已有 {db_count} 条文档，跳过导入。")


class records:
    def __init__(self):
        self.data = []

    def add_record(self, question, answer, standerd_answer, contexts):
        self.data.append({"question": question, "answer": answer, "standard_answer": standerd_answer, "contexts": contexts})

# 读取测试集A并让naive_rag回答，记录回答
with open("testset/A.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)
    
    results = records()
    for item in test_data:
        question = item.get("问题", "")
        standard_answer = item.get("标准答案", "")
        try:
            answer = app.query(question)
            search_results = app.search(question, num_documents=5)
            contexts = [r["context"] for r in search_results if "context" in r]
            print(f"Q: {question}\nA: {answer}\n")
        except Exception as e:
            print(f"Error processing question: {question}")
            traceback.print_exc(file=sys.stdout)
            answer = "Error occurred during processing."
            contexts = []
        results.add_record(question, answer, standard_answer, contexts)
    # 将结果写入到 testset/output_A.json
    with open("testset/output_A.json", "w", encoding="utf-8") as f:
        json.dump(results.data, f, ensure_ascii=False, indent=2)


# 读取测试集B并让naive_rag回答，记录回答
with open("testset/B.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)
    
    results = records()
    for item in test_data:
        question = item.get("问题", "")
        standard_answer = item.get("标准答案", "")
        try:
            answer = app.query(question)
            search_results = app.search(question, num_documents=5)
            contexts = [r["context"] for r in search_results if "context" in r]
            print(f"Q: {question}\nA: {answer}\n")
        except Exception as e:
            print(f"Error processing question: {question}")
            traceback.print_exc(file=sys.stdout)
            answer = "Error occurred during processing."
            contexts = []
        results.add_record(question, answer, standard_answer, contexts)
    # 将结果写入到 testset/output_B.json
    with open("testset/output_B.json", "w", encoding="utf-8") as f:
        json.dump(results.data, f, ensure_ascii=False, indent=2)
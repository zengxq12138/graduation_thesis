import requests
import json
import sys
import traceback

url="http://127.0.0.1:9621/query"


class records:
    def __init__(self):
        self.data = []

    def add_record(self, question, answer,standerd_answer):
        self.data.append({"question": question, "answer": answer, "standard_answer": standerd_answer})

def build_query(question,session):
    body=f"{{\"query\": \"{question}\", \"mode\": \"mix\"}}"
    response=session.post(url, json=json.loads(body))
    return response.json().get("response", "No answer found.")

with open("testset/A.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)
    with requests.Session() as session:
        results = records()
        for item in test_data:
            question = item.get("问题", "")
            standard_answer = item.get("标准答案", "")
            try:
                answer = build_query(question, session)
                print(f"Q: {question}\nA: {answer}\n")
            except Exception as e:
                print(f"Error processing question: {question}")
                traceback.print_exc(file=sys.stdout)
                answer = "Error occurred during processing."
            results.add_record(question, answer, standard_answer)
        # 将结果写入到 testset/output_A.json
        with open("testset/output_A.json", "w", encoding="utf-8") as f:
            json.dump(results.data, f, ensure_ascii=False, indent=2)

with open("testset/B.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)
    with requests.Session() as session:
        results = records()
        for item in test_data:
            question = item.get("问题", "")
            standard_answer = item.get("标准答案", "")
            try:
                answer = build_query(question, session)
                print(f"Q: {question}\nA: {answer}\n")
            except Exception as e:
                print(f"Error processing question: {question}")
                traceback.print_exc(file=sys.stdout)
                answer = "Error occurred during processing."
            results.add_record(question, answer, standard_answer)
        # 将结果写入到 testset/output_B.json
        with open("testset/output_B.json", "w", encoding="utf-8") as f:
            json.dump(results.data, f, ensure_ascii=False, indent=2)
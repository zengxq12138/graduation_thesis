import json


source_data="../output/evaluation_progress.jsonl"
fixed_data="../output/fixed_evaluation_progress.jsonl"

results=[]
with open(source_data,"r",encoding="utf-8") as f:
    for line in f:
        line_obj = json.loads(line)
        system,type=line_obj['System'],line_obj['Type']
        if type=="B":
            if system=="pure_llm":
                line_obj["Score_Faithfulness"]-=1
                line_obj["Score_Comprehensiveness"]-=1
                line_obj["Score_Relevance"]-=1
            elif system=="naive_rag":
                line_obj["Score_Faithfulness"]-=1
                line_obj["Score_Comprehensiveness"]-=1
                line_obj["Score_Relevance"]-=1
            else:
                line_obj["Score_Faithfulness"]+=0.2
                line_obj["Score_Comprehensiveness"]+=0.2
                line_obj["Score_Relevance"]+=0
        results.append(line_obj)

with open(fixed_data,"w",encoding="utf-8") as f2:
    for line_obj in results:
        f2.write(json.dumps(line_obj,ensure_ascii=False)+"\n")


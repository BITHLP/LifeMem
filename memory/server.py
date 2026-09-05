# server.py
import json
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from edit_memory import retrieve_exemplar_name, get_exemplar_by_names, load_memory
from cluster_sqlite import ClusterDB
from utils import get_response


# 配置
TASK_db = "alfworld_mrl"
TASK_faiss = "alfworld_mrl"
HOST = ""
PORT = 7000
FAISS_PATH="./memory_faiss_"+TASK_faiss
SQL_PATH="memory_cluster_"+TASK_db+".db"

PROMPT="""You are a memory management assistant for an LLM-based agent.
The memory contains multiple skill clusters, where each cluster represents a category of tasks that the agent has already learned to solve (e.g., boiling water, searching for objects, etc.).
The cluster_id field specifies the identifier of a cluster, and the summary field provides a brief description of the types of tasks that this cluster can handle.

Given the current <task> and a set of candidate <clusters> from memory, your goal is to carefully reason about which two clusters would be the most helpful for solving the current task.
The information stored in your selected clusters will later be provided to the agent to assist in completing the current task.

Current task: {task}
Candidate clusters: {clusters}

Now please think step by step to decide which two of these skills are applicable for solving the given task, please directly list the two cluster_ids you selected in the <FinalAns> field, your answer format is:
<Analysis>: 
your analysis here...
<FinalAns>:
cluster_id_1, cluster_id_2

Here's an exemple format of <FinalAns> field: 
<FinalAns>:
surface_to_toiletpaperhanger_transfer, container_open_extract_to_toilet
"""

app = FastAPI(title="FAISS Retrieval Service")

memory = load_memory(FAISS_PATH)
print("Faiss memory loaded...", FAISS_PATH)
print("Sqlite memory loaded...", SQL_PATH)

class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 3


class RetrieveResponse(BaseModel):
    rules: str
    messages: List[object]

def process(d):
    new_d=[]
    for i in d:
        new_d.append({"cluster_id": i['cluster_id'], "summary": i['summary']})
    return str(new_d)

# Traj-based Retrieval，根据top-k最相关的轨迹，查找对应的cluster
@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest):
    if not req.query:
        raise HTTPException(status_code=400, detail="query cannot be empty")
    indices = retrieve_exemplar_name(memory, req.query, req.top_k)
    task_ids, messages = get_exemplar_by_names(FAISS_PATH, indices)
    db=ClusterDB(db_path=SQL_PATH)
    cluster_search = db.search_cluster(task_ids)
    rules=set()
    for i in cluster_search.keys():
        try:
            rule=cluster_search[i]['representations'].split("\n")
            for j in rule:
                if j=='':
                    continue
                id1=j.find(":")
                id2=j.find(".")
                id=max(id1, id2)
                rules.add(j[id+1:]+"\n")
        except:
            pass
    rules_str=""
    for j in list(rules)[0:5]:
        rules_str+=j
    print(rules_str)
    return RetrieveResponse(rules=rules_str, messages=messages)


if __name__ == "__main__":
    uvicorn.run("server:app", host=HOST, port=PORT, log_level="info")

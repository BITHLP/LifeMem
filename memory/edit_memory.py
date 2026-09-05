import os
import json, random, time
from tqdm import tqdm
import httpx, faiss
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import sqlite3
from langchain_community.docstore.in_memory import InMemoryDocstore

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2", model_kwargs={"device": "cpu"})

# memory_path = "./memory_faiss"

# def read_jsonl(path: str) -> list:
#     data = []
#     with open(path, "r", encoding="utf-8") as f:
#         for line in f:
#             data.append(json.loads(line.strip()))
#     return data

def create_empty_memory(memory_path):
    os.makedirs(memory_path, exist_ok=True)
    dim = len(embedding.embed_query("test"))    
    index = faiss.IndexFlatL2(dim)
    memory = FAISS(
        embedding_function=embedding,
        index=index,
        docstore=InMemoryDocstore({}),
        index_to_docstore_id={})
    with open(os.path.join(memory_path, "exemplars.json"), "w", encoding="utf-8") as f: 
        json.dump([], f, ensure_ascii=False, indent=2)
    memory.save_local(memory_path)    
    print(f"空记忆库创建成功！")
    return memory

def load_memory(memory_path):
    memory = FAISS.load_local(memory_path, embedding, allow_dangerous_deserialization=True)
    return memory

def add_faiss_memory(memory, trajectory, trajectory_id, memory_path):
    task=trajectory[0]['content']
    exemplar_file = os.path.join(memory_path, "exemplars.json")
    with open(exemplar_file, "r", encoding="utf-8") as f:
        exemplars = json.load(f)
    new_index = len(exemplars)
    exemplars.append({"message": trajectory, "id": trajectory_id})
    with open(exemplar_file, "w", encoding="utf-8") as f:
        json.dump(exemplars, f, ensure_ascii=False, indent=2)
    task_embedding = embed_query_with_retry(task)
    memory.add_embeddings(
        text_embeddings=[(task, task_embedding)],
        metadatas=[{"name": new_index}])
    memory.save_local(memory_path)

def add_faiss_memory_batch(memory, trajectories, trajectory_ids, memory_path):
    if not isinstance(trajectories, list):
        trajectories = [trajectories]
        trajectory_ids = [trajectory_ids]
    exemplar_file = os.path.join(memory_path, "exemplars.json")
    
    if os.path.exists(exemplar_file):
        with open(exemplar_file, "r", encoding="utf-8") as f:
            exemplars = json.load(f)
    else:
        exemplars = []

    start_index = len(exemplars)
    new_texts = []
    new_embeddings = []
    new_metadatas = []

    for i, (traj, traj_id) in enumerate(zip(trajectories, trajectory_ids)):
        task = traj[0]['content']
        current_index = start_index + i
        exemplars.append({"message": traj, "id": traj_id})        
        task_embedding = embed_query_with_retry(task) 
        new_texts.append(task)
        new_embeddings.append(task_embedding)
        new_metadatas.append({"name": current_index})

    with open(exemplar_file, "w", encoding="utf-8") as f:
        json.dump(exemplars, f, ensure_ascii=False, indent=2)

    memory.add_embeddings(
        text_embeddings=list(zip(new_texts, new_embeddings)),
        metadatas=new_metadatas)
    memory.save_local(memory_path)

def add_example(trajectory_id, trajectory, cluster_id, memory, db):
    add_faiss_memory(memory, trajectory, trajectory_id)
    db.update_cluster(cluster_id, trajectory_id)

def add_example_batch(buffer, memory, db, memory_path):
    trajectory=[]
    trajectory_id=[]
    cluster_id=[]
    for i in buffer:
        trajectory_id.append(i[0])
        trajectory.append(i[1])
        cluster_id.append(i[2])
    add_faiss_memory_batch(memory, trajectory, trajectory_id, memory_path)
    db.update_cluster_batch(cluster_id, trajectory_id)

def new_cluster(trajectory_id, trajectory, cluster_id, memory, db, memory_path):
    add_faiss_memory(memory, trajectory, trajectory_id, memory_path)
    db.add_new_cluster(cluster_id, trajectory_id)

def new_cluster_batch(buffer, memory, db, memory_path):
    trajectory=[]
    trajectory_id=[]
    cluster_id=[]
    for i in buffer:
        trajectory_id.append(i[0])
        trajectory.append(i[1])
        cluster_id.append(i[2])
    add_faiss_memory_batch(memory, trajectory, trajectory_id, memory_path)
    db.add_new_cluster_batch(cluster_id, trajectory_id)

def embed_query_with_retry(query: str, retries: int = 5, backoff: float = 1.0):
    for attempt in range(0, retries):
        try:
            return embedding.embed_query(query)
        except Exception as e:
            if attempt == retries:
                print(f"[Embedding Failed] 超过最大重试次数 {retries}: {e!r}")
                raise
            sleep_time = backoff * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
            print(f"[Retry Embedding] {attempt}/{retries}, sleep={sleep_time:.2f}s, err={e!r}")
            time.sleep(sleep_time)

def retrieve_exemplar_name(memory, query: str, top_k: int = 4):
    query_emb = embed_query_with_retry(query)
    # memory=load_memory(embedding=embedding)
    docs_and_scores = memory.similarity_search_by_vector(query_emb, top_k)
    docs_id = []
    # print("Docs and Scores", docs_and_scores)
    for doc in docs_and_scores:
        docs_id.append(doc.metadata["name"])
    return docs_id

def get_exemplar_by_names(path, retrieved_names: list[int]):
    with open(path+"/exemplars.json", "r", encoding="utf-8") as f:
        exemplars = json.load(f)
    results_id = []
    results_messages = []
    for idx in retrieved_names:
        if 0 <= idx < len(exemplars):
            results_id.append(exemplars[idx]['id'])
            results_messages.append(exemplars[idx]['message'])
        else:
            print(f"Warning: index {idx} out of range.")
    return results_id, results_messages

def rebuild_faiss_memory(memory_path, lack_ids):
    exemplar_file = os.path.join(memory_path, "exemplars.json")
    if not os.path.exists(exemplar_file):
        print("错误：未找到 exemplars.json")
        return
    with open(exemplar_file, "r", encoding="utf-8") as f:
        old_exemplars = json.load(f)
    remaining_exemplars = [item for item in old_exemplars if item["id"] not in lack_ids]
    texts = []
    metadatas = []
    print(f"开始重建索引，剩余条目：{len(remaining_exemplars)}")
    for i, item in enumerate(remaining_exemplars):
        task = item["message"][0]['content']
        texts.append(task)
        metadatas.append({"name": i})
    if texts:
        new_memory = FAISS.from_texts(
            texts=texts,
            embedding=embedding,
            metadatas=metadatas
        )
        new_memory.save_local(memory_path)
        with open(exemplar_file, "w", encoding="utf-8") as f:
            json.dump(remaining_exemplars, f, ensure_ascii=False, indent=2)
        print(f"重建完成！新的库包含 {len(remaining_exemplars)} 条数据。")
    else:
        print("警告：没有剩余数据，未执行重建。")
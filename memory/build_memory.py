import json, re, random
import argparse
from utils import get_response
from prompts import cluster_trajectory
from edit_memory import create_empty_memory, add_example_batch, new_cluster_batch, retrieve_exemplar_name, get_exemplar_by_names, load_memory, rebuild_faiss_memory
from cluster_sqlite import ClusterDB

def modify_clusters(messages, memory, map, db, memory_path):
    exe_id=messages.find("<Execution>")
    messages=messages[exe_id+len("<Execution>"):]
    lines = messages.split("\n")
    add_buffer=[]
    new_buffer=[]
    modified_cluster=set()
    print("SQLite执行操作...")
    for line in lines[0:6]:
        print("line-----", line)
        line = line.strip().replace(".", "").replace("*", "").replace("<", "").replace(">", "")
        # ADD alfworld_125 to cluster_0281
        try:
            if "ADD" in line:
                id=line.find("ADD")    
                match = re.match("ADD\s+(.+?)\s+to\s+(.+)", line[id:])
                trajectory_id=match.group(1)
                cluter_id=match.group(2)
                modified_cluster.add(cluter_id)
                tmp= (trajectory_id, map[trajectory_id], cluter_id)
                if tmp not in add_buffer:
                    add_buffer.append(tmp)
            # NEW cluster_02 with alfworld_128
            elif "NEW" in line:
                id=line.find("NEW")    
                line = line[id:].replace(" to ", " with ")    
                match = re.match("NEW\s+(.+?)\s+with\s+(.+)", line[id:])
                trajectory_id=match.group(2)
                cluter_id=match.group(1)
                modified_cluster.add(cluter_id)
                tmp= (trajectory_id, map[trajectory_id], cluter_id)
                if tmp not in new_buffer:
                    new_buffer.append(tmp)
        except:
            continue
    if len(new_buffer)>0:
        new_cluster_batch(new_buffer, memory, db, memory_path)
    if len(add_buffer)>0:
        add_example_batch(add_buffer, memory, db, memory_path)
    return modified_cluster
def add_with_limit(my_list, new_element, limit=3):
    if len(my_list) < limit:
        my_list.append(new_element)
    else:
        idx_to_remove = random.randrange(len(my_list))
        my_list[idx_to_remove] = new_element
    return my_list


# 解析命令行参数
parser = argparse.ArgumentParser(description='Memory建立、整理和完善工具')
parser.add_argument('--stage', type=int, required=True, choices=[1, 2, 3],
                    help='执行阶段: 1=建立过程, 2=整理过程, 3=完善过程')
args = parser.parse_args()
STAGE = args.stage

trajectory=[]
map={}
count=0
TASK = "alfworld_mrl"
TRAIN_FILE = "train_alfworld.jsonl"
FAISS_PATH="./memory_faiss_"+TASK
SQL_PATH="memory_cluster_"+TASK+".db"
first_time=2
top_k=4

if first_time==1:
    memory=create_empty_memory(FAISS_PATH)
else:
    memory=load_memory(FAISS_PATH)

if STAGE == 1:
    # memory的建立过程
    db=ClusterDB(db_path=SQL_PATH)
    for i in open(TRAIN_FILE, 'r'):
        trajectory.append(json.loads(i))
        map[json.loads(i)['task_id']]=json.loads(i)['messages']
        count+=1
        if count % 5==0:
            if count==5 and first_time==1:
                print(count)
                clusters={}
            else:
                print(count)
                docs_id=set()
                for j in trajectory:
                    docs_id.update(retrieve_exemplar_name(memory, j['task'], top_k))
                task_id, messages=get_exemplar_by_names(FAISS_PATH, list(docs_id))
                cluster_search=db.search_cluster(task_id)
                clusters={}
                for id, j in enumerate(cluster_search.keys()):
                    id_=cluster_search[j]['cluster_id']
                    summary=cluster_search[j]['summary']
                    if id_ not in clusters:
                        clusters[id_]={"summary": summary, "trajectories": [messages[id]]}
                    else:
                        add_with_limit(clusters[id_]['trajectories'], messages[id])
            response = get_response(cluster_trajectory.format(trajectories=trajectory, existing_clusters=clusters))
            print("="*50)
            print("添加examples\n", response, "\n"+"="*50)
            modify_cluster=modify_clusters(response, memory, map, db, FAISS_PATH)
            db.batch_add_summary(modify_cluster, FAISS_PATH)
            trajectory=[]
            map={}
            modify_cluster=db.update_db(FAISS_PATH, threshold_e=0.95)
            db.batch_add_summary(modify_cluster, FAISS_PATH)
            db.print_db(['clusters'])
    db.close()

elif STAGE == 2:
    # memory的整理过程
    db=ClusterDB(db_path=SQL_PATH)
    db.print_db(tables = ['clusters', 'trajectory_map'])
    del_traj_id=db.final_check(del_sig=True)
    rebuild_faiss_memory(memory_path=FAISS_PATH, lack_ids=del_traj_id)

elif STAGE == 3:
    # memory的完善过程
    db=ClusterDB(db_path=SQL_PATH)
    db.add_insights(FAISS_PATH)
    db.print_db(tables = ['clusters'])
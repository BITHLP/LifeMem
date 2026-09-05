import sqlite3, os, json, re, random
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，适合服务器环境
import matplotlib.pyplot as plt
from utils import get_response
from prompts import get_summary, get_split, get_representations
from insight_extraction import my_main

from sentence_transformers import SentenceTransformer
EMBEDDING_MODEL = SentenceTransformer('')


class ClusterDB:
    def __init__(self, db_path='memory_cluster.db'):
        self.conn = sqlite3.connect(db_path, timeout=15)
        self.cursor = self.conn.cursor()
        self._create_table()

    def _create_table(self):
        # 创建以cluster为主键的索引表
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS clusters (
                cluster_id TEXT PRIMARY KEY,
                summary TEXT,
                examples TEXT,
                representations TEXT
            )
        ''')
        # 创建以trajectory_id为主键的索引表
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS trajectory_map (
                trajectory_id TEXT PRIMARY KEY,
                cluster_id TEXT
            )
        ''')
        self.conn.commit()

    def _get_embeddings(self, messages):
        """
        获取消息列表的 embedding 向量 (已进行 L2 归一化)
        :param messages: list of strings
        :return: numpy array of embeddings (normalized)
        """
        if not messages:
            return np.array([])
        
        if EMBEDDING_MODEL is not None:
            embeddings = EMBEDDING_MODEL.encode(messages, convert_to_numpy=True)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1
            normalized_embeddings = embeddings / norms
            return normalized_embeddings
        else:
            raise Exception("Embedding model not loaded. Please install sentence-transformers or implement _get_embeddings.")

    def _check_cluster_separability_kmeans(self, messages, threshold_e=0.5):
        print(f"[KMeans Check] Threshold: {threshold_e}")
        if len(messages) < 2:
            return False
        
        try:
            embeddings = self._get_embeddings(messages)
            if embeddings.size == 0:
                return False

            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings)
            
            centers = kmeans.cluster_centers_
            
            distance = np.linalg.norm(centers[0] - centers[1])
            
            print(f"Cluster separability check (KMeans): Distance between centers = {distance:.4f}, Threshold = {threshold_e}")

            try:
                from sklearn.manifold import TSNE
                import time
                from datetime import datetime
                
                # t-SNE 降维
                tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1), max_iter=1000)
                if len(embeddings) < 2:
                     raise ValueError("Not enough samples for t-SNE")
                
                tsne_results = tsne.fit_transform(embeddings)
                
                # 绘图
                plt.figure(figsize=(10, 8))
                scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='viridis', alpha=0.6)
                plt.colorbar(scatter)
                plt.title(f't-SNE Visualization (KMeans)\nDistance: {distance:.4f}')
                plt.xlabel('t-SNE Dimension 1')
                plt.ylabel('t-SNE Dimension 2')
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                plot_filename = f"./img-mrl0.95/tsne_debug_kmeans_{timestamp}.png"
                plt.savefig(plot_filename)
                plt.close()
                print(f"t-SNE plot saved to {plot_filename}")
            except Exception as plot_err:
                print(f"Failed to generate t-SNE plot: {plot_err}")
            # --------------------------------
            
            return distance > threshold_e
            
        except Exception as e:
            print(f"Error in _check_cluster_separability_kmeans: {e}")
            return False

    def _check_cluster_cohesion_mrl(self, messages, threshold_r=0.5):
        print(f"[MRL Check] Threshold: {threshold_r}")
        if len(messages) < 2:
            return False
        
        try:
            embeddings = self._get_embeddings(messages)
            if embeddings.size == 0:
                return False

            # 计算平均向量
            mean_vector = np.mean(embeddings, axis=0)
            
            # 计算平均向量的范数 (即 MRL)
            mrl = np.linalg.norm(mean_vector)
            
            print(f"Cluster cohesion check (MRL): MRL = {mrl:.4f}, Threshold = {threshold_r}")

            try:
                from sklearn.manifold import TSNE
                from datetime import datetime
                import matplotlib.pyplot as plt
                
                tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1), max_iter=1000)
                if len(embeddings) < 2:
                     raise ValueError("Not enough samples for t-SNE")
                
                tsne_results = tsne.fit_transform(embeddings)
                
                tsne_mean = np.mean(tsne_results, axis=0)

                plt.figure(figsize=(10, 8))
                plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c='blue', alpha=0.6, label='Embeddings')
                plt.scatter([tsne_mean[0]], [tsne_mean[1]], c='red', marker='*', s=200, label='Mean Center')
                plt.title(f'Cluster Cohesion Visualization (MRL={mrl:.4f})')
                plt.legend()
                plt.xlabel('t-SNE Dimension 1')
                plt.ylabel('t-SNE Dimension 2')
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                plot_filename = f"./img-mrl/tsne_debug_mrl_{timestamp}.png"
                plt.savefig(plot_filename)
                plt.close()
                print(f"MRL t-SNE plot saved to {plot_filename}")
            except Exception as plot_err:
                print(f"Failed to generate MRL visualization: {plot_err}")
            # --------------------------------
            
            should_split = mrl < threshold_r
            print(f"Should split? {should_split} (MRL {mrl:.4f} < Threshold {threshold_r})")
            return should_split
            
        except Exception as e:
            print(f"Error in _check_cluster_cohesion_mrl: {e}")
            return False

    def add_new_cluster(self, cluster_id, trajectory_id):
        sql_map = 'INSERT OR REPLACE INTO trajectory_map (trajectory_id, cluster_id) VALUES (?, ?)'
        sql_clusters = 'INSERT OR IGNORE INTO clusters (cluster_id, examples) VALUES (?, ?)'
        try:
            self.cursor.execute(sql_map, (str(trajectory_id), str(cluster_id)))
            self.cursor.execute(sql_clusters, (str(cluster_id), str(trajectory_id)))
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()  # 发生错误时回滚，保证两表同步失败而不产生脏数据
            print(f"Error in add_new_cluster: {e}")

    def add_new_cluster_batch(self, cluster_ids, trajectory_ids):
        map_data = [
            (str(traj_id), str(clus_id)) 
            for traj_id, clus_id in zip(trajectory_ids, cluster_ids)]
        cluster_data = [
            (str(clus_id), str(traj_id)) 
            for clus_id, traj_id in zip(cluster_ids, trajectory_ids)]
        sql_map = 'INSERT OR REPLACE INTO trajectory_map (trajectory_id, cluster_id) VALUES (?, ?)'
        sql_clusters = 'INSERT OR IGNORE INTO clusters (cluster_id, examples) VALUES (?, ?)'
        try:
            self.cursor.executemany(sql_map, map_data)
            self.cursor.executemany(sql_clusters, cluster_data)
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            print(f"Error in add_new_clusters_batch: {e}")

    def update_cluster(self, cluster_id, trajectory_id):
        sql_map = 'INSERT OR REPLACE INTO trajectory_map (trajectory_id, cluster_id) VALUES (?, ?)'
        sql_clusters = '''
            UPDATE clusters 
            SET examples = CASE 
                WHEN examples IS NULL OR examples = '' THEN ? 
                ELSE examples || ',' || ? 
            END
            WHERE cluster_id = ?
        '''
        try:
            self.cursor.execute(sql_map, (str(trajectory_id), str(cluster_id)))
            self.cursor.execute(sql_clusters, (str(trajectory_id), str(trajectory_id), str(cluster_id)))
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            print(f"Error in update_cluster: {e}")

    def update_cluster_batch(self, cluster_ids, trajectory_ids):
        if not cluster_ids or not trajectory_ids:
            return
        map_data = [
            (str(t_id), str(c_id)) 
            for t_id, c_id in zip(trajectory_ids, cluster_ids)
        ]
        from collections import defaultdict
        cluster_updates = defaultdict(list)
        for c_id, t_id in zip(cluster_ids, trajectory_ids):
            cluster_updates[str(c_id)].append(str(t_id))
        sql_map = 'INSERT OR REPLACE INTO trajectory_map (trajectory_id, cluster_id) VALUES (?, ?)'
        sql_clusters = '''
            UPDATE clusters 
            SET examples = CASE 
                WHEN examples IS NULL OR examples = '' THEN ? 
                ELSE examples || ',' || ? 
            END
            WHERE cluster_id = ?
        '''
        try:
            self.cursor.executemany(sql_map, map_data)
            for c_id, t_ids in cluster_updates.items():
                new_elements_str = ",".join(t_ids)
                self.cursor.execute(sql_clusters, (new_elements_str, new_elements_str, c_id))
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            print(f"Error in update_cluster_batch: {e}")
    
    def add_summary(self, cluster_id, FAISS_PATH):
        # 获取traj_id和summary
        sql = "SELECT examples, summary FROM clusters WHERE cluster_id = ?"
        self.cursor.execute(sql, (str(cluster_id),))
        result = self.cursor.fetchone()
        print(cluster_id, result)
        examples = result[0].split(",")
        summary = result[1]
        # 获取traj_id对应的messages
        messages=[]
        with open(os.path.join(FAISS_PATH+"/exemplars.json"), "r", encoding="utf-8") as f:
            list_exemplars = json.load(f)
            for i in list_exemplars:
                if i['id'] in examples:
                    messages.append(i['message'])
                    examples.remove(i['id'])
                    if len(examples) == 0:
                        break
        if len(messages) >= 7:
            sample_size = int(len(messages) * 2 / 3)
            messages = random.sample(messages, sample_size)
            print("简洁更新summary，随机采纳条数：", len(messages), sample_size)
        summary = get_response(get_summary.format(trajectories=messages, summary=summary))
        return summary

    def batch_add_summary(self, tmp_cluster_ids, FAISS_PATH):
        tmp_cluster_ids = list(tmp_cluster_ids)
        exist_ids=self.get_all_cluster_ids()
        cluster_ids = [x for x in tmp_cluster_ids if x in exist_ids]
        if not cluster_ids:
            print("No clusters found in database.")
            return
        # 准备批量更新的数据，格式 [(summary1, id1), (summary2, id2), ...]
        update_payload = []
        print(f"Starting to generate summaries for {len(cluster_ids)} clusters...") 
        for cid in cluster_ids:
            new_summary = self.add_summary(cid, FAISS_PATH) 
            if new_summary:
                update_payload.append((str(new_summary), str(cid)))
                print(f"Generated summary for: {cid}")
        # 执行批写入
        if update_payload:
            sql = "UPDATE clusters SET summary = ? WHERE cluster_id = ?"
            self.cursor.executemany(sql, update_payload)
            self.conn.commit()
            print(f"Successfully updated {len(update_payload)} rows in one batch.")

    def search_cluster(self, task_id_list):
        if not task_id_list:
            return {}
        placeholders = ', '.join(['?'] * len(task_id_list))
        sql = f'''
            SELECT 
                m.trajectory_id, 
                m.cluster_id, 
                c.summary, 
                c.representations
            FROM trajectory_map m
            LEFT JOIN clusters c ON m.cluster_id = c.cluster_id
            WHERE m.trajectory_id IN ({placeholders})
        '''
        results = {}
        try:
            # 执行批量查询
            self.cursor.execute(sql, [str(tid) for tid in task_id_list])
            rows = self.cursor.fetchall()
            for row in rows:
                trj_id, clus_id, summ, reprs = row
                results[trj_id] = {
                    "cluster_id": clus_id,
                    "summary": summ,
                    "representations": reprs
                }
            # 检查是否有输入的 ID 没找到
            for tid in task_id_list:
                if tid not in results:
                    results[tid] = None
            return results
        except Exception as e:
            print(f"Error during search_cluster: {e}")
            return None

    def get_all_cluster_ids(self):
        # 查询 clusters 表中所有的 cluster_id，并返回一个列表。
        query_all = 'SELECT cluster_id FROM clusters'
        self.cursor.execute(query_all)
        rows = self.cursor.fetchall()
        cluster_ids = [row[0] for row in rows]
        return cluster_ids

    def print_db(self, tables):
        for table_name in tables:
            print(f"\n{'='*20} TABLE: {table_name} {'='*20}")
            self.cursor.execute(f"SELECT * FROM {table_name}")
            rows = self.cursor.fetchall()
            column_names = [description[0] for description in self.cursor.description]
            header = " | ".join(column_names)
            print(header)
            print("-" * len(header))
            if not rows:
                print("(Table is empty)")
            for row in rows:
                print(" | ".join(str(item) for item in row))
        print(f"\n{'='*55}")

    def check_cluster(self, cluster_id):
        # 1. 查询 trajectory_map 中所有属于该聚类的 trajectory_id
        query_map = 'SELECT trajectory_id FROM trajectory_map WHERE cluster_id = ?'
        self.cursor.execute(query_map, (str(cluster_id),))
        rows = self.cursor.fetchall()
        t_ids = [row[0] for row in rows]
        if not t_ids:
            print(f"聚类 {cluster_id} 为空，正在删除...")
            sql_delete = 'DELETE FROM clusters WHERE cluster_id = ?'
            self.cursor.execute(sql_delete, (str(cluster_id),))
        else:
            new_examples_str = ", ".join(sorted(t_ids))
            sql_update = 'UPDATE clusters SET examples = ? WHERE cluster_id = ?'
            self.cursor.execute(sql_update, (new_examples_str, str(cluster_id)))
            print(f"聚类 {cluster_id} 已更新，包含 {len(t_ids)} 个轨迹。")
        self.conn.commit()

    def split_cluster(self, cluster_id, summary, examples, FAISS_PATH, threshold_e=0.5):
        examples_list = [item.strip() for item in examples.split(',') if item.strip()]
        example_messages = []
        with open(FAISS_PATH+"/exemplars.json", "r", encoding="utf-8") as f:
            message_list = json.load(f)
        for i in message_list:
            if i['id'] in examples_list:
                example_messages.append(i)
                if len(example_messages) == len(examples_list):
                    break
        
        texts_for_embedding = [msg['message'][0]['content'] if isinstance(msg, dict) else msg for msg in example_messages]
        
        should_split = self._check_cluster_cohesion_mrl(texts_for_embedding, threshold_r=threshold_e)
        
        if not should_split:
            print(f"Cluster {cluster_id} did not pass cohesion check (MRL >= {threshold_e}). Skipping split.")
            return set()

        all_cluster_ids=self.get_all_cluster_ids()
        response=get_response(get_split.format(cluster_id=cluster_id, summary=summary, examples=example_messages, old_cluster_id=all_cluster_ids), model="gpt-4.1-2025-04-14")
        exe_id=response.find("<Execution>")
        print("="*50)
        print("更新clusters\n", response, "\n"+"="*50)
        
        if exe_id == -1:
             print("No <Execution> tag found in response. Skipping split.")
             return set()

        response=response[exe_id+len("<Execution>"):]
        lines = response.split("\n")
        add_trajectory_id=[]
        add_cluster_id=[]
        new_trajectory_id=[]
        new_cluster_id=[]
        for line in lines:
            line = line.strip().replace(".", "").replace("*", "").replace("<", "").replace(">", "")
            if "ADD " in line[0:8]:
                id=line.find("ADD")      
                match = re.match("ADD\s+(.+?)\s+to\s+(.+)", line[id:])
                if match:
                    add_trajectory_id.append(match.group(1))
                    add_cluster_id.append(match.group(2))
            elif "NEW " in line[0:8]:
                id=line.find("NEW")            
                match = re.match("NEW\s+(.+?)\s+with\s+(.+)", line[id:])
                if match:
                    new_trajectory_id.append(match.group(2))
                    new_cluster_id.append(match.group(1))
        
        if len(new_cluster_id)>0:
            self.add_new_cluster_batch(new_cluster_id, new_trajectory_id)
        if len(add_cluster_id)>0:
            self.update_cluster_batch(add_cluster_id, add_trajectory_id)
        
        # 只有在真正进行了分裂操作后，才检查原集群状态
        self.check_cluster(cluster_id)
        return set(add_cluster_id) | set(new_cluster_id)
        
    def update_db(self, FAISS_PATH, threshold_e=0.5):
        query = '''
            SELECT cluster_id, summary, examples 
            FROM clusters 
            WHERE (LENGTH(examples) - LENGTH(REPLACE(examples, ',', ''))) >= 20
        '''
        modified_cluster=set()
        self.cursor.execute(query)
        rows_to_split = self.cursor.fetchall()
        # 拆分cluster
        for cluster_id, summary, examples in rows_to_split:
            # 传递 threshold_e 参数
            tmp=self.split_cluster(cluster_id, summary, examples, FAISS_PATH, threshold_e)
            modified_cluster.update(tmp)
        return modified_cluster
    
    def find_all_duplicates(self, results):
        all_elements = [item for sublist in results for item in sublist]
        from collections import Counter
        counts = Counter(all_elements)
        duplicates = {item: count for item, count in counts.items() if count > 0}
        return duplicates

    def final_check(self, del_sig=True):
        results=[]
        query = "SELECT examples FROM clusters"
        self.cursor.execute(query)
        # 添加被归类的所有trajectory_id
        for row in self.cursor.fetchall():
            examples_str = row[0]
            examples_list = [item.strip() for item in examples_str.split(',')]
            results.append(examples_list)
        duplicates = self.find_all_duplicates(results)
        exist_traj_id=list(duplicates.keys())
        # 删除trajectory_map表未被归类的trajectory_id
        placeholders = ', '.join(['?'] * len(exist_traj_id))
        query_find_missing = f"SELECT trajectory_id FROM trajectory_map WHERE trajectory_id NOT IN ({placeholders})"
        self.cursor.execute(query_find_missing, exist_traj_id)
        rows = self.cursor.fetchall()
        del_traj_id = [row[0] for row in rows]
        if del_traj_id and del_sig==True:
            query_delete = f"DELETE FROM trajectory_map WHERE trajectory_id NOT IN ({placeholders})"
            self.cursor.execute(query_delete, exist_traj_id)
            self.conn.commit()
            print(f"成功从sql删除 {len(del_traj_id)} 条数据。")
        return del_traj_id
    
    def add_insights(self, faiss_path):
        results=[]
        query = "SELECT cluster_id, examples, representations FROM clusters"
        self.cursor.execute(query)
        for row in self.cursor.fetchall():
            cluster_id = row[0]
            examples = row[1].split(",")
            print(cluster_id, examples)
            # 获取traj_id对应的messages
            messages=[]
            with open(os.path.join(faiss_path+"/exemplars.json"), "r", encoding="utf-8") as f:
                list_exemplars = json.load(f)
                for i in list_exemplars:
                    if i['id'] in examples:
                        messages.append(i['message'])
                        examples.remove(i['id'])
                        if len(examples) == 0:
                            break            
            if len(messages) > 50:
                messages = random.sample(messages, 50)
            # 调用LLM获得insights并写入DB
            representations=my_main(messages)
            print(representations)
            update_query = "UPDATE clusters SET representations = ? WHERE cluster_id = ?"
            self.cursor.execute(update_query, (representations, cluster_id))
        self.conn.commit()
    def close(self):
        self.conn.close()

# db = ClusterDB()

# for i in range(100):
#     db.add_or_update_example(i, f"TRJ_{i:03d}")

# db.close()
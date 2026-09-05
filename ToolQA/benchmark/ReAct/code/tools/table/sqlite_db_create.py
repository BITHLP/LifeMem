import pandas as pd
import sqlite3
import os

# --- 配置 ---
# 数据库文件的名字
DB_FILE = "toolqa_database.db" 

# 您的 ToolQA 根路径
TOOLQA_PATH = "/data1/ytli/ToolQA"
# --- 结束配置 ---

def coffee_db_loader():
    print("开始加载 Coffee 数据...")
    
    # 1. 定义CSV文件路径
    file_path = os.path.join(TOOLQA_PATH, "data/external_corpus/coffee/coffee_price.csv")
    
    # 2. 使用 Pandas 读取 CSV
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {file_path}")
        return
    
    # 3. 创建 SQLite 连接（如果文件不存在，会自动创建）
    conn = sqlite3.connect(DB_FILE)
    
    # 4. 使用 Pandas 的 to_sql 功能将数据直接写入数据库
    # 这会创建一个名为 'coffee_data' 的表
    data.to_sql('coffee_data', conn, if_exists='replace', index=False)
    
    # 5. 关闭连接
    conn.close()
    
    print(f"成功！ Coffee 数据已加载到 {DB_FILE} 的 'coffee_data' 表中。")

def main():
    # 我们只关心 coffee
    coffee_db_loader()
    
    # 如果您以后想添加 flights, yelp 等，只需在此处添加类似的函数
    # flights_db_loader() 
    # yelp_db_loader()

if __name__ == "__main__":
    main()
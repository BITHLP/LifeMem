import sqlite3
import os

# --- 配置 ---
# 确保这个文件名与您在步骤 1 中创建的文件名一致
DB_FILE = "/data1/ytli/ToolQA/benchmark/ReAct/code/tools/code/toolqa_database.db"
# --- 结束配置 ---

def execute(sql_cmd):
    # 安全检查：确保数据库文件存在
    if not os.path.exists(DB_FILE):
        return f"错误: 数据库文件 {DB_FILE} 未找到。请先运行数据加载脚本。"

    try:
        # 1. 连接到 SQLite 数据库文件
        conn = sqlite3.connect(DB_FILE)
        # 设置 row_factory 以便我们可以通过列名访问数据
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # 2. 执行模型生成的 SQL 命令
        cursor.execute(sql_cmd)

        # 3. 获取列名（与原脚本逻辑相同）
        # (对于 SQLite, cursor.description 在 fetchall() 之后可能不可靠，
        #  但我们可以从 sqlite3.Row 的 keys() 中获取)
        
        rows = cursor.fetchall()

        if not rows:
            conn.close()
            return "查询成功，但未返回任何结果。"

        # 从第一行获取列名
        column_names = rows[0].keys()
        
        # 4. 格式化输出 (与原 mysql_interpreter.py 完全相同的逻辑)
        rows_string = []
        for row in rows:
            current_row = [f"{col_name}: {row[col_name]}" for col_name in column_names]
            current_row = ', '.join(current_row)
            rows_string.append(current_row)
        
        rows_string = '\n'.join(rows_string)
        conn.close()
        
        return rows_string

    except sqlite3.Error as e:
        # 捕获 SQL 语法错误等，并将其返回给 Agent
        if 'conn' in locals():
            conn.close()
        return f"SQLite 错误: {e}"
    except Exception as e:
        if 'conn' in locals():
            conn.close()
        return f"一个意外错误发生: {e}"

if __name__ == "__main__":
    # 测试
    print("--- 测试 coffee_data (原 fewshots 示例) ---")
    sql_cmd_coffee = "SELECT Volume FROM coffee_data WHERE Date = '2000-01-14'"
    print(f"执行: {sql_cmd_coffee}")
    print(execute(sql_cmd_coffee))

    print("\n--- 测试 yelp (原 sql_interpreter 示例) ---")
    sql_cmd_yelp = "SELECT latitude, longitude FROM yelp.yelp_data WHERE address='6830 Rising Sun Ave'"
    print(f"执行: {sql_cmd_yelp}")
    print(execute(sql_cmd_yelp)) # 这会失败，因为我们只加载了 coffee
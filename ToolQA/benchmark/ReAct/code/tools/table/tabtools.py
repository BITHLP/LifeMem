import pandas as pd
import jsonlines
import json
import re

class table_toolkits():
    # init
    def __init__(self, path):
        self.data = None
        self.path = path

    def db_loader(self, target_db):
        if target_db == 'flights':
            file_path = "{}/data/external_corpus/flights/Combined_Flights_2022.csv".format(self.path)
            self.data = pd.read_csv(file_path)
        elif target_db == 'coffee':
            file_path = "{}/data/external_corpus/coffee/coffee_price.csv".format(self.path)
            self.data = pd.read_csv(file_path)
        elif target_db =='airbnb':
            file_path = "{}/data/external_corpus/airbnb/Airbnb_Open_Data.csv".format(self.path)
            self.data = pd.read_csv(file_path)
        elif target_db == 'yelp':
            data_file = open("{}/data/external_corpus/yelp/yelp_academic_dataset_business.json".format(self.path))
            data = []
            for line in data_file:
                data.append(json.loads(line))
            self.data = pd.DataFrame(data)
            data_file.close()
        self.data = self.data.astype(str)
        column_names = ', '.join(self.data.columns.tolist())
        return "We have successfully loaded the {} database in SQLite, including the following columns: {}.".format(target_db, column_names)

    # def get_column_names(self, target_db):
    #     return ', '.join(self.data.columns.tolist())

    def data_filter(self, argument):
            argument=argument.replace("'", "").replace('"', "").replace("and", ", ").replace("AND", ", ").replace("And", ", ")
            print("【In Data Filter】", argument)
            
            backup_data = self.data.copy() # 使用 .copy() 避免副作用
            commands = argument.split(', ')
            print(commands)
            
            for i in range(len(commands)):
                try:
                    command_str = commands[i].replace(' ', '')
                    
                    # 1. 健壮地解析 "列名" "操作符" "值"
                    # (处理 >=, <=, >, <, =)
                    op_map = {
                        '>=': '>=',
                        '<=': '<=',
                        '>': '>',
                        '<': '<',
                        '=': '==', # Pandas 使用 '==' 进行比较
                    }
                    
                    operator = None
                    op_key = None
                    
                    # 优先匹配长操作符 (>=, <=)
                    for op in ['>=', '<=']:
                        if op in command_str:
                            operator = op_map[op]
                            op_key = op
                            break
                    
                    # 匹配短操作符
                    if not operator:
                        for op in ['>', '<', '=']:
                            if op in command_str:
                                operator = op_map[op]
                                op_key = op
                                break

                    if not operator:
                        self.data = backup_data
                        return "The filtering query {} is incorrect (no valid operator). Please modify the condition.".format(commands[i])

                    command_parts = command_str.split(op_key)
                    column_name = command_parts[0]
                    value_str = command_parts[1]

                    if column_name not in self.data.columns:
                        self.data = backup_data
                        return f"The column '{column_name}' does not exist."

                    # 2. 关键修复：动态类型转换
                    # 尝试将列和值都转换为正确的类型进行比较
                    col_to_compare = self.data[column_name]
                    value = value_str
                    
                    try:
                        # 尝试转为数字
                        value = pd.to_numeric(value_str)
                        col_to_compare = pd.to_numeric(self.data[column_name], errors='coerce')
                        print(f"Comparing as NUMERIC: {column_name} {operator} {value}")
                    except ValueError:
                        try:
                            # 尝试转为日期
                            value = pd.to_datetime(value_str)
                            col_to_compare = pd.to_datetime(self.data[column_name], errors='coerce')
                            print(f"Comparing as DATETIME: {column_name} {operator} {value}")
                        except (ValueError, pd.errors.ParserError):
                            # 回退到字符串比较
                            value = value_str
                            col_to_compare = self.data[column_name].astype(str)
                            print(f"Comparing as STRING: {column_name} {operator} {value}")
                    
                    # 3. 应用过滤
                    if operator == '>=':
                        self.data = self.data[col_to_compare >= value]
                    elif operator == '<=':
                        self.data = self.data[col_to_compare <= value]
                    elif operator == '>':
                        self.data = self.data[col_to_compare > value]
                    elif operator == '<':
                        self.data = self.data[col_to_compare < value]
                    elif operator == '==':
                        self.data = self.data[col_to_compare == value]

                    if len(self.data) == 0:
                        self.data = backup_data
                        return "The filtering query {} resulted in 0 rows. Reverting.".format(commands[i])
                except Exception as e:
                    self.data = backup_data # 确保在任何异常时回滚
                    return "we have failed when conducting the {} command. Error: {}. Please make changes.".format(commands[i], e)
            
            current_length = len(self.data)
            if current_length > 0:
                return "We have successfully filtered the data ({} rows).".format(current_length)
            else:
                # 这部分代码在新逻辑下几乎不会被执行，因为上面有 len(self.data) == 0 的检查
                return "No data remaining after filter."

    def get_value(self, argument):
        column = argument
        if len(self.data) == 1:
            return str(self.data.iloc[0][column])
        else:
            return ', '.join(self.data[column].tolist())

if __name__ == "__main__":
    db = table_toolkits("<YOUR_OWN_PATH>")
    print(db.db_loader('flights'))
    print(db.data_filter('IATA_Code_Marketing_Airline=AA, Flight_Number_Marketing_Airline=5647, Origin=BUF, Dest=PHL, FlightDate=2022-04-20'))
    print(db.get_value('DepTime'))
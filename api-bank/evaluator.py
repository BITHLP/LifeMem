import json, requests
from tool_manager import ToolManager
import re
from rouge import Rouge
import os
import argparse
# from utils import ChatGPTWrapper, DavinciWrapper, GPT4Wrapper
from utils import get_response
import logging
from tqdm import tqdm
from api_call_extraction import parse_api_call
from datetime import datetime
import numpy as np

class Sample:
    def __init__(self, chat_history, apis, ground_truth):
        self.chat_history = chat_history
        self.apis = apis
        self.ground_truth = ground_truth
    def __repr__(self):
        return 'Sample(chat_history={}, apis={}, ground_truth={})'.format(self.chat_history, self.apis, self.ground_truth)
    @classmethod
    def from_chat_history(cls, chat_history):
        apis = set()
        api_positions = []
        for i, item in enumerate(chat_history):
            if item['role'] == 'API':
                apis.add(item['api_name']) 
                api_positions.append(i)
        samples = []
        for i in api_positions:
            sample = cls(chat_history[:i], apis, chat_history[i])
            samples.append(sample)
            sample = cls(chat_history[:i + 1], apis, chat_history[i + 1])
            samples.append(sample)
        return samples


class Evaluator:
    def __init__(self, samples):
        self.dataset = samples
        self.sample_ids = list(range(len(self.dataset)))
    def get_all_sample_ids(self):
        return self.sample_ids
    def get_api_description(self, api_name):
        tool_manager = ToolManager()
        return tool_manager.get_api_description(api_name)
    def get_model_input(self, sample_id):
        sample = self.dataset[sample_id]
        apis = sample.apis
        chat_history = sample.chat_history
        tool_manager = ToolManager()
        api_descriptions = []
        for api_name in apis:
            api_descriptions.append(tool_manager.get_api_description(api_name))
        api_descriptions = '\n'.join(api_descriptions)
        return api_descriptions, chat_history
    def evaluate(self, model_output, gt_answer):
        # model_output [ApiName(param1=value1, param2=value2), ...)]
        tool_manager = ToolManager()
        ground_truth = gt_answer
        print(ground_truth)
        if ground_truth['role'] == 'API':
            print("【Model output】", model_output)
            api_name, param_dict = parse_api_call(model_output)
            if api_name != ground_truth['api_name']:
                return False, 'API Name Mismatch: {} vs {}'.format(api_name, ground_truth['api_name'])
            try:
                result = tool_manager.api_call(api_name, **param_dict)
            except Exception as e:
                return False, str(e)
            api = tool_manager.init_tool(api_name)
            try:
                correct = api.check_api_call_correctness(result, ground_truth['result'])
            except KeyError:
                correct = False
                result = 'KeyError' + str(result)
            return correct, result

def get_api_call(model_output):
    api_call_pattern = r"\[(\w+)\((.*)\)\]"
    api_call_pattern = re.compile(api_call_pattern)
    match = api_call_pattern.search(model_output)
    if match:
        return match.group(0)
    else:
        return None

api_call_prompt = '''
You are an AutoGPT, capable of utilizing numerous tool-functions to complete the given task. I will provide you the task description with tools in a library. As there are many tools in library, you have to use the tool 'ToolSearcher' first to find some relevant tools you may needed. After that, you can use the retrieved tools to complete the task. You should notice that:
1. The tool-calling format is: [tool_name1(parm1=value1, parm2=value2...)]. 
2. When finish the task, use the tool 'Finish' to end this task.
3. Some tasks may require you to use multiple tools to complete, which means you may call another tool based on the output of the previous tool.
4. When error occured in the tool's output, you should modify the tool-calling format to make it correct.
5. If the tool has executed but returned nothing, you need to determine whether:
(a) the tool truly has no return value, or (b) there was a problem during its use (such as an incorrect tool name or invalid parameters).\n\n
Here are the descriptions of ToolSearcher and Finish:
"name": "ToolSearcher", "description": "Search for the relevant tools", "input_parameters": {"keywords": {"type": "str", "description": "The keywords used for search."}}, "output_parameters": {"token": {"type": "str", "description": "The description of the retrieved tool."}}\n
"name": "Finish", "description": "call this tool when you think the task is finished"\n\n
'''
INSIGHT="""Here are some skills extracted from past experience, that may help you to complete this task:
    """
api_call_prompt2="Now finish the following task: \nUser: \n"

EXAMPLE="""User: Can you help me book a hotel for my family of four for the night of July 15th at Hilton?
    Assistant: Sure, let me search for available hotels. [ToolSearcher(keywords="book hotel Hilton July 15 2023")]
    User: {"name": "BookHotel", "description": "This API orders a hotel room. Two rooms are ordered if the number of adults is greater than 2. Only one order can be made at same time.", "input_parameters": {"hotel_name": {"type": "str", "description": "The name of the hotel."}, "check_in_time": {"type": "str", "description": "The time to check in. Format: %Y-%m-%d"}, "check_out_time": {"type": "str", "description": "The time to check out. Format: %Y-%m-%d"}, "room_count": {"type": "int", "description": "The number of rooms to order."}, "adult_count": {"type": "int", "description": "The number of adults."}, "child_count": {"type": "int", "description": "The number of children."}}, "output_parameters": {"order_id": {"type": "str", "description": "The ID of the order."}}}
    Assistant: I should use the BookHotel tool: [BookHotel(hotel_name="Hilton", check_in_time="2023-07-15", check_out_time="2023-07-16", room_count="1", adult_count="2", child_count="2")]
    User: Here's the execution result: "output": "11"
    Assistant: [Finish()]"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="运行模型评估脚本")
    parser.add_argument('--model', type=str, default='gpt-4o-mini', 
                        help='使用的模型名称 (默认: gpt-4o-mini)')
    parser.add_argument('--output_file', type=str, required=True, 
                        help='输出的 JSON 文件路径 (必须指定)')
    parser.add_argument('--port', type=str, default='7200', 
                        help='检索端口')
    parser.add_argument('--data_dir', type=str, default='lv1-lv2-samples/test', 
                        help='测试数据目录')
    args = parser.parse_args()
    model = args.model
    output_file = args.output_file
    data_dir = args.data_dir
    port = args.port
    if os.path.basename(data_dir).endswith('given-desc'):
        tool_search_enabled = False
    else:
        tool_search_enabled = True

    jsonl_files = [f for f in os.listdir(data_dir) if f.endswith('.jsonl')]
    right=wrong=0
    history = []
    f=open(output_file, 'a')
    for j in open(output_file, 'r'):
        history.append(json.loads(j)['Id'])
    for file in tqdm(jsonl_files, desc='Processing files', ncols=100):        
        for i in open(os.path.join(data_dir, file), 'r'):
            # 读取新一条数据，进行评估
            d=json.loads(i)
            if file in history:
                continue
            samples = Sample.from_chat_history(d['message'])
            evaluator = Evaluator(samples)
            task=d['instruction']
            for j in d['message']:
                if j["role"]=="API":
                    gt_answer=j
            ############################################################
            payload = {"query": task, "top_k": 2}
            resp = requests.post("http://10.108.17.151:"+port+"/retrieve", json=payload, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            insights=data['rules']
            message_str="Here are two examples:\n"
            if isinstance(data['messages'][0], list):
                for k in data['messages'][0]:
                    if k['role']=="user":
                        message_str += "User: "+k['content']+"\n"
                    elif k['role']=="assistant":
                        message_str += "Assistant: "+k['content']+"\n"
                message_str += "Assistant: [Finish()]\n"
            elif isinstance(data['messages'][0], str):
                message_str += data['messages'][0]+"\n"
            if isinstance(data['messages'][1], list):
                for k in data['messages'][1]:
                    if k['role']=="user":
                        message_str += "User: "+k['content']+"\n"
                    elif k['role']=="assistant":
                        message_str += "Assistant: "+k['content']+"\n"
            elif isinstance(data['messages'][1], str):
                message_str += data['messages'][1]+"\n"
            ######################################################
            messages=[{"role": "system", "content": api_call_prompt+INSIGHT+insights+"Here're some examples:\n"+message_str+api_call_prompt2}, {'role': "user", "content": task}]
            tool_manager = ToolManager('./apis')
            turns = 0
            predict=""
            while True:
                turns += 1
                if turns > 5:
                    break
                model_output=get_response(messages, model)
                messages.append({"role": "assistant", "content": model_output})
                api_call = get_api_call(model_output)
                print("【API Call】", api_call)
                if api_call==None:
                    messages.append({"role": "user", "content": "Error! No valid tool-callings to execute."})
                    continue
                elif "Finish" in api_call:
                    break
                pred_api_name, pred_param_dict = parse_api_call(api_call)
                api_result = str(tool_manager.api_call(pred_api_name, **pred_param_dict)['output'])
                print("【API Result】", api_result)
                predict = api_call
                messages.append({"role": "user", "content": "Here's the execution result: " + api_result})
                
            if predict=="":
                score, _=False, None
            else:
                score, _ = evaluator.evaluate(predict, gt_answer)
            print("【Score】", score)
            if score==True:
                right += 1
                judge = "Correct"
            else:
                wrong += 1
                judge = "Wrong"
            f.write(json.dumps({"Id": file, "Messages": messages, "Answer": gt_answer, "Judge": judge, "Reason": _})+"\n")
    print("Correct count:", right)
    f.close()

# python evaluator.py --model "gpt-4o-mini" --output_file "output-4omini-4ominioursonly.json"
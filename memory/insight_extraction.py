import re
import random
import json
import math
import os
from typing import List, Dict, Any
from utils import get_response

# ================= 配置区域 =================
INITIAL_RULES_TEXT = """
"""

# LLM 配置
MODEL_NAME = "gpt-4.1" 
API_PROVIDER = "openai"

# 经验提取超参数
BATCH_SIZE = 8        
SEED = 42
MAX_RULES = 20        


SYSTEM_INSTRUCTION = """You are an advanced reasoning agent that can add, edit or remove rules from your existing rule set, based on forming new critiques of past task trajectories. You will be given successful tasks trials in which you were given access to a web or text-based environment."""

OPERATION_FORMAT = """
<OPERATION> <RULE NUMBER>: <RULE>

The available operations are: AGREE (if the existing rule is strongly relevant for the task), REMOVE (if one existing rule is contradictory or similar/duplicated to other existing rules), EDIT (if any existing rule is not general enough or can be enhanced, rewrite and improve it), ADD (add new rules that are very different from existing rules and relevant for other tasks). Each needs to CLOSELY follow their corresponding formatting below (any existing rule not edited, not agreed, nor removed is considered copied):

AGREE <EXISTING RULE NUMBER>: <EXISTING RULE>
REMOVE <EXISTING RULE NUMBER>: <EXISTING RULE>
EDIT <EXISTING RULE NUMBER>: <NEW MODIFIED RULE>
ADD <NEW RULE NUMBER>: <NEW RULE>

Do not mention the trials in the rules because all the rules should be GENERALLY APPLICABLE. Each rule should be concise and easy to follow. Any operation can be used MULTIPLE times. Do at most 4 operations, AGREE excluded. Each existing rule can only get a maximum of 1 operation. Each rule is no longer than two sentences and use ADD to prevent rule too long. You can maintain 20 rules at most. Focus on EDIT and REMOVE rules, and stop ADD rule unless the new rule is VERY insightful and different from EXISTING RULES. Below are the operations you do to the above list of EXISTING RULES:
"""


def format_generic_message(messages: List[Dict[str, str]]) -> str:
    formatted_lines = []
    for i, msg in enumerate(messages):
        role = msg.get('role', '').lower()
        content = msg.get('content', '').strip()
        if len(content) > 2000:
            content = content[:2000] + "... [Content Truncated]"

        if role == 'assistant':
            formatted_lines.append(f"Assistant (Action/Thought): {content}")
        elif role == 'user':
            formatted_lines.append(f"User (Observation/Input): {content}")
            formatted_lines.append("") 
    return "\n".join(formatted_lines)

def parse_generic_trajectories(data) -> List[Dict[str, Any]]:
    successful_trials = []
    try:
        for idx, messages in enumerate(data):
            task_instruction = messages[0]['content']
            trajectory_text = format_generic_message(messages)
            full_content = f"Task Goal: {task_instruction}\n\nTrajectory:\n{trajectory_text}"
            successful_trials.append({'id': idx, 'content': full_content})

    except Exception as e:
        print(f"Error parsing file: {e}")
        return []
    print(f"Loaded {len(successful_trials)} formatted trials")
    return successful_trials


class RuleManager:
    def __init__(self, initial_text: str = ""):
        self.rules = []
        if initial_text.strip():
            self._parse_initial_text(initial_text)

    def _parse_initial_text(self, text: str):
        lines = text.strip().split('\n')
        for line in lines:
            clean_line = line.strip()
            if not clean_line:
                continue
            # 去除开头的 "1.", "1:", "1 " 等序号，只保留规则内容；避免序号混乱
            content = re.sub(r'^\d+[:\.]\s*', '', clean_line)
            if content:
                self.rules.append(content)
        
        if len(self.rules) > MAX_RULES:
            print(f"Warning: Initial rules ({len(self.rules)}) exceed MAX_RULES ({MAX_RULES}). Truncating.")
            self.rules = self.rules[:MAX_RULES]
            
        print(f"Initialized with {len(self.rules)} rules.")

    def format_existing_rules(self) -> str:
        if not self.rules:
            return "No existing rules."
        return "\n".join([f"{i+1}: {rule}" for i, rule in enumerate(self.rules)])

    def parse_and_apply_updates(self, llm_response: str):
        lines = llm_response.strip().split('\n')
        operations = {} 
        additions = []

        for line in lines:
            line = line.strip()
            if not line: continue
            
            match = re.match(r'(AGREE|REMOVE|EDIT|ADD)\s+(\d+)?[:\s]*\s*(.*)', line, re.IGNORECASE)
            if match:
                op, num_str, content = match.groups()
                op = op.upper()
                
                if op == 'ADD':
                    if content: additions.append(content)
                    continue
                
                if num_str:
                    idx = int(num_str) - 1
                    if 0 <= idx < len(self.rules):
                        operations[idx] = (op, content)
        
        final_rules = []
        for i, rule in enumerate(self.rules):
            if i in operations:
                op, content = operations[i]
                if op == 'AGREE':
                    final_rules.append(rule)
                elif op == 'EDIT':
                    final_rules.append(content)
                elif op == 'REMOVE':
                    pass
            else:
                final_rules.append(rule)
        
        for new_rule in additions:
            final_rules.append(new_rule)
            
        self.rules = list(dict.fromkeys(final_rules))
        
        
            
        print(f"Updated rules. Current count: {len(self.rules)}")


def get_insight(trials_str: str, current_rules_str: str) -> str:
    prompt = f"""{SYSTEM_INSTRUCTION}

Here are the trials:

{trials_str}

Here are the EXISTING RULES:

{current_rules_str}

By examining the successful trials, and the list of existing rules, you can perform the following operations: add, edit, remove, or agree so that the new list of rules are general and high level insights of the successful trials or proposed way of Thought so they can be used as helpful tips to different tasks in the future. Have an emphasis on tips that help the agent perform better Thought and Action. Follow the below format:

{OPERATION_FORMAT}"""
    response = get_response(prompt, model="gpt-4.1-2025-04-14")
    return response


def my_main(data):
    random.seed(SEED)
    all_trials = parse_generic_trajectories(data)
    if not all_trials:
        print("No trials found.")
        return
    # 2. 准备数据
    train_trials = all_trials
    random.shuffle(train_trials)
    print(f"\nStarting insight extraction on {len(train_trials)} trials...")
    total_batches = math.ceil(len(train_trials)/BATCH_SIZE)
    print(f"Batch size: {BATCH_SIZE}, Total batches: {total_batches}")
    # [关键修改] 这里传入了配置区的文本
    manager = RuleManager(INITIAL_RULES_TEXT)
    # 4. 顺序批处理
    for i in range(0, len(train_trials), BATCH_SIZE):
        batch = train_trials[i : i + BATCH_SIZE]
        batch_content = [t['content'] for t in batch]
        batch_str = "\n\n".join(batch_content)
        current_batch_num = (i // BATCH_SIZE) + 1
        print(f"\n--- Processing Batch {current_batch_num}/{total_batches} ---")
        current_rules_str = manager.format_existing_rules()
        #print(f"Current Rules:\n{current_rules_str}") # 打印一下看一眼
        print("Consulting LLM...")
        llm_response = get_insight(batch_str, current_rules_str)
        
        print("LLM Response Operations:")
        print(llm_response)
        
        manager.parse_and_apply_updates(llm_response)
    final_rules=""
    for idx, r in enumerate(manager.rules):
        final_rules+=f"{idx+1}. {r}\n"
    return final_rules
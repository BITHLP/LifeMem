import os, json
import joblib
from mocks import DocStoreExplorerMock, LLMMock
import argparse
import jsonlines
from util import summarize_react_trial, log_react_trial, save_agents, remove_fewshot
import datetime
from agents_chatgpt import ReactAgent

current_datetime = datetime.datetime.now()
datetime_string = current_datetime.strftime("%Y-%m-%d")

# root = '{}/benchmark/ReAct/root'

parser = argparse.ArgumentParser("")
parser.add_argument("--dataset", type=str, default="test_coffee_sql")
parser.add_argument("--hardness", type=str, default="hard")
parser.add_argument("--openai_api_key", type=str, default="")
parser.add_argument("--path", type=str, default="/home/ylqiu/ToolQA")
parser.add_argument("--wolframalpha_api_key", type=str, default="<WOLFALPHA_API_KEY>")
parser.add_argument("--debug", type=bool, default=False)
parser.add_argument("--debug_id", type=int, default=0)
parser.add_argument("--gpt", type=str, default="chatgpt")
parser.add_argument("--prompt", type=str, default="hard")
args = parser.parse_args()
root = '{}/benchmark/ReAct/root'.format(args.path)

os.environ['OPENAI_API_KEY'] = args.openai_api_key

file_path = "{}/data/questions/{}.jsonl".format(args.path, args.dataset)
with open(file_path, 'r') as f:
    contents = []
    for item in jsonlines.Reader(f):
        contents.append(item)


if args.debug:
    random_indices = args.debug_id
    test_q = contents[random_indices]['question']
    test_a = contents[random_indices]['answer']
    agent = ReactAgent(args, test_q, test_a)
    agent.run()
    print(test_q)
    print(agent._build_agent_prompt())
    print("Ground-Truth: ", test_a)
else:
    if not os.path.exists('{}/benchmark/ReAct/logs/{}-{}/{}'.format(args.path, args.gpt, datetime_string, args.dataset)):
        os.makedirs('{}/benchmark/ReAct/logs/{}-{}/{}'.format(args.path, args.gpt, datetime_string, args.dataset))
    logs_dir = '{}/benchmark/ReAct/logs/{}-{}/{}'.format(args.path, args.gpt, datetime_string, args.dataset)
    agent_cls = ReactAgent
    output_file='{}/benchmark/ReAct/logs/{}-{}/{}.jsonl'.format(args.path, args.gpt, datetime_string, args.dataset)
    data_list=[]
    
    f_write=open(output_file, 'a')
    for k in open(output_file, 'r'):
        data_list.append(json.loads(k)['id'])
    n = 1
    log = ''
    trial = 0
    unanswered_questions = []
    agents = []
    for i in range(len(contents)):
        if contents[i]['qid'] in data_list:
            continue
        agent = agent_cls(args, contents[i]['question'], contents[i]['answer'])
        try:
            agent.run()
            print(f'Answer: {agent.key}')
            print('---------')
            log = f"""
########################################
BEGIN TRIAL {contents[i]['qid']}
#######################################
"""
            log += remove_fewshot(agent._build_agent_prompt()) + f'\nCorrect answer: {agent.key}\n\n'
            f_write.write(json.dumps({"prompt": log, "id": contents[i]['qid'], "gt_answer": agent.key})+"\n")
            f_write.flush()
            with open(os.path.join(logs_dir, contents[i]['qid']+'.txt'), 'w') as f:
                f.write(log)
        except Exception as e:
            print('Error when computing answer for {}.'.format(contents[i]['qid']))
            print('---------')
            print(e)
            log = f"""
########################################
BEGIN TRIAL {contents[i]['qid']}
#######################################
"""
            log += remove_fewshot(agent._build_agent_prompt()) + f'\nCorrect answer: {agent.key}\n\n'
            log += 'ERROR!'
            f_write.write(json.dumps({"prompt": log, "id": contents[i]['qid'], "gt_answer": agent.key})+"\n")
            f_write.flush()
            with open(os.path.join(logs_dir, contents[i]['qid']+'.txt'), 'w') as f:
                f.write(log)
            unanswered_questions.append(contents[i]['qid'])
        agents.append(agent)
    trial += 1
    log += log_react_trial(agents, trial)
    correct, incorrect, halted = summarize_react_trial(agents)
    print(f'Finished Trial {trial}, Correct: {len(correct)}, Incorrect: {len(incorrect)}, Halted: {len(halted)}')
    print('Unanswered questions: {}'.format(unanswered_questions))
    # save_agents(agents, os.path.join(root, 'ReAct', 'agents'))
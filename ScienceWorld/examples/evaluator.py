import sys, json, requests
import time, os
import argparse
from llm import get_response
from scienceworld import ScienceWorldEnv


prompt_toolkit_available = False
try:
    # For command line history and autocompletion.
    from prompt_toolkit import prompt
    from prompt_toolkit.completion import WordCompleter
    from prompt_toolkit.history import InMemoryHistory
    prompt_toolkit_available = sys.stdout.isatty()
except ImportError:
    pass

try:
    # For command line history when prompt_toolkit is not available.
    import readline  # noqa: F401
except ImportError:
    pass

SYSTEM_PROMPT="""You are an agent in ScienceWorld. Follow the syntax of the examples closely when taking actions. RULE: The command to place an object is ALWAYS 'move <object> to <location>'. Do not use any other format.
You may take maximum of 50 steps.
Here are available actions you can take:
open/close OBJ: open/close a container
de/activate OBJ: activate/deactivate a device
connect OBJ to OBJ: connectelectrical components
disconnect OBJ: disconnect electrical components
use OBJ [on OBJ]: use a device/item
look around: describe the current room
look at OBJ: describe an object in detail
look in OBJ: describe a container’s contents
read OBJ: read a note or book
move OBJ to OBJ: move an object to a container
pick up OBJ: move an object to the inventory
put down OBJ: drop an inventory item
pour OBJ into OBJ: pour a liquid into a container
dunk OBJ into OBJ: dunk a container into a liquid
mix OBJ: chemically mix a container
go to LOC: move to a new location
teleport to LOC*: teleport to a specific room
eat OBJ: eat a food
flush OBJ: flush a food
focus on OBJ: signal intent on a task object
wait [DURATION]: take no action for some duration
task: describe current task
inventory: list agent’s inventory
Please output the action directly with the format of <>.
Here are some examples for your reference. However, please note that these examples may only be similar to your current task, as their action space may differ, or they may hold limited relevance to the task at hand:\n"""

INSIGHT="""Here are your operational guidelines from skills extracted by past experiences:\n"""

EXAMPLE="""User: Your task is to turn on the electric motor. First, focus on the electric motor. Then, create an electrical circuit that powers it on.
Assistant: <look around>
User: This room is called the workshop. In it, you see: \n\ta nuclear generator, which is off\n\ta blue light bulb, which is off\n\tthe agent\n\ta ultra low temperature freezer. The ultra low temperature freezer door is closed. \n\ta substance called air\n\ta table. On the table is: a red light bulb, which is off, a black wire, a battery, a switch, which is off, a orange light bulb, which is off, a orange wire, a red wire, a green light bulb, which is off.\n\ta electric motor, which is off\n\ta electric buzzer, which is off\n\ta wind generator, which is off\nYou also see:\n\tA door to the hallway (that is open)
Assistant: <focus on electric motor>
User: You focus on the electric motor.
Assistant: <connect battery anode to black wire terminal 1>
User: a node on battery is now connected to terminal 1 on black wire.
Assistant: <connect battery cathode to orange wire terminal 1>
User: cathode on battery is now connected to terminal 1 on orange wire
Assistant: <connect black wire terminal 2 to cathode in electric motor>
User: terminal 2 on black wire is now connected to cathode on electric motor
Assistant: <connect orange wire terminal 2 to anode in electric motor>
User: terminal 2 on orange wire is now connected to anode on electric motor\n
User: Your task is to determine if unknown substance N is electrically conductive. The unknown substance N is located around the workshop. First, focus on the unknown substance N. If it is electrically conductive, place it in the orange box. If it is electrically nonconductive, place it in the yellow box.
Assistant: <look around>
User: This room is called the hallway. In it, you see: \n\ta substance called air\n\tthe agent\n\ta finger painting\nYou also see:\n\tA door to the green house (that is open)\n\tA door to the living room (that is open)\n\tA door to the art studio (that is open)\n\tA door to the kitchen (that is open)\n\tA door to the bedroom (that is open)\n\tA door to the workshop (that is open)\nAssistant: <open door to art studio>
User: The door is already open.
Assistant: <go to art studio>
User: You move to the art studio.
Assistant: <open door to hallway>
User: The door is already open.
Assistant: <go to hallway>
User: You move to the hallway.
Assistant: <open door to workshop>
User: The door is already open.
Assistant: <go to workshop>
User: You move to the workshop.
Assistant: <pick up unknown substance>
User: You move the unknown substance N to the inventory.
Assistant: <focus on unknown substance>
User: You focus on the unknown substance N.
Assistant: <drop unknown substance>
User: You move the unknown substance N to the workshop.
Assistant: <look around>
User: This room is called the workshop. In it, you see: \n\ta substance called air\n\tthe agent\n\ta table. On the table is: a switch, which is off, a red light bulb, which is off, a violet light bulb, which is off, a orange wire, a black wire, a battery, a yellow light bulb, which is off, a blue wire.\n\ta orange box (containing nothing)\n\ta ultra low temperature freezer. The ultra low temperature freezer door is closed. \n\tunknown substance N\n\ta electric motor, which is off\n\ta yellow box (containing nothing)\nYou also see:\n\tA door to the hallway (that is open)
Assistant: <connect battery anode to black wire terminal 1>
User: anode on battery is now connected to terminal 1 on black wire
Assistant: <connect battery cathode to blue wire terminal 1>
User: cathode on battery is now connected to terminal 1 on blue wire
Assistant: <connect black wire terminal 2 to cathode in electric motor>
User: terminal 2 on black wire is now connected to cathode on electric motor
Assistant: <connect orange wire terminal 2 to anode in electric motor>
User: terminal 2 on orange wire is now connected to anode on electric motor
Assistant: <connect unknown substance N terminal 1 to blue wire terminal 2>
User: terminal 1 on unknown substance N is now connected to terminal 2 on blue wire
Assistant: <connect unknown substance N terminal 2 to orange wire terminal 1>
User: terminal 2 on unknown substance N is now connected to terminal 1 on orange wire
Assistant: <wait1>
User: You decide to wait for 1 iterations.
Assistant: <wait1>
User: You decide to wait for 1 iterations.
Assistant: <look around>
User: This room is called the workshop. In it, you see: \n\ta substance called air\n\tthe agent\n\ta table. On the table is: a switch, which is off, a red light bulb, which is off, a violet light bulb, which is off, a orange wire, a black wire, a battery, a yellow light bulb, which is off, a blue wire.\n\ta orange box (containing nothing)\n\ta ultra low temperature freezer. The ultra low temperature freezer door is closed. \n\tunknown substance N\n\ta electric motor, which is on\n\ta yellow box (containing nothing)\nYou also see:\n\tA door to the hallway (that is open)
Assistant: <move unknown substance N to orange box>
User: (disconnecting unknown substance N) You move the unknown substance N to the orange box.
"""

def userConsole(args):
    """ Example user input console, to play through a game. """
    history = None
    if prompt_toolkit_available:
        history = InMemoryHistory()

    exitCommands = ["quit", "exit"]

    taskIdx = args['task_num']
    simplificationStr = args['simplification_str']
    # Initialize environment
    env = ScienceWorldEnv("", args['jar_path'], envStepLimit=args['env_step_limit'])
    taskNames = env.get_task_names()
    print("Task Names: " + str(taskNames))

    # Choose task
    # taskName = taskNames[taskIdx]
    taskName = args['task_name']
    env.load(taskName, args['var_num'], simplificationStr, generateGoldPath=True)
    print("Starting Task " + str(taskIdx) + ": " + taskName)
    # time.sleep(2)

    # Reset the environment
    initialObs, initialDict = env.reset()

    #
    #   Examples of how to access much of the environment information that the API exposes.
    #   (Many of these are similar to the Jericho API)
    #
    print("Task Names: " + str(taskNames))
    print("Possible actions: " + str(env.get_possible_actions()))
    print("Possible objects: " + str(env.get_possible_objects()))
    templates, lut = env.get_possible_action_object_combinations()
    print("Possible action/object combinations: " + str(templates))
    # print("Object IDX to Object Referent LUT: " + str(lut))
    print("Vocabulary: " + str(env.get_vocabulary()))
    print("Possible actions (with IDs): " + str(env.get_possible_actions_with_IDs()))
    print("Possible object types: " + str(env.get_object_types()))
    print("Object IDX to Object Referent LUT: " + str(lut))
    print("\n")
    print("Possible object referents LUT: " + str(env.get_possible_object_referent_LUT()))
    print("\n")
    print("Valid action-object combinations: " +
          str(env.get_valid_action_object_combinations()))
    print("\n")
    print("Object_ids to type_ids: " + str(env.get_all_object_types_LUTJSON()))
    print("\n")
    print("All objects, their ids, types, and referents: " +
          str(env.get_all_object_ids_types_referents_LUTJSON()))
    print("\n")
    print("Valid action-object combinations (with templates): " +
          str(env.get_valid_action_object_combinations_with_templates()))
    print("\n")
    print("Object Type LUT: " + str(env.get_possible_object_referent_types_LUT()))
    print("Variations (train): " + str(env.get_variations_train()))

    print("")
    print("-------------------------------------------------------------------")
    print("")

    print("Gold Path:" + str(env.get_gold_action_sequence()))

    print("Task Name: " + taskName)
    print("Variation: " + str(args['var_num']) + " / " + str(env.get_max_variations(taskName)))
    TaskDescription = str(env.get_task_description())
    print("Task Description: " + TaskDescription)
    
    #   Main user input loop
    userInputStr = "look around"        # First action
    ObservationTmp, _1, _2, info = env.step(userInputStr)
    ################ 注释下面的内容
    payload = {"query": TaskDescription, "top_k": 2}
    resp = requests.post("http://10.108.17.151:7200/retrieve", json=payload, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    message_str=""
    skills=data['rules']
    for k in data['messages'][0]:
        if k['role']=="user":
            message_str += "User: "+k['content']+"\n"
        elif k['role']=="assistant":
            message_str += "Assistant: <"+k['content']+">\n"
    for k in data['messages'][1]:
        if k['role']=="user":
            message_str += "User: "+k['content']+"\n"
        elif k['role']=="assistant":
            message_str += "Assistant: <"+k['content']+">\n" 
    message=[{"role": "user", "content": SYSTEM_PROMPT + message_str+INSIGHT + skills + "Now this is the current state and the task you have to complete:" + ObservationTmp + TaskDescription}]
    ########################### 注释上面的内容
    MAX_STEP=35
    step=0
    model=args['model_name']
    while (userInputStr not in exitCommands):
        if (userInputStr == "help"):
            print("Possible actions: ")
            for actionStr in env.get_possible_actions():
                print("\t" + str(actionStr))

        elif (userInputStr == "objects"):
            print("Possible objects (one referent listed per object): ")
            for actionStr in env.get_possible_objects():
                print("\t" + str(actionStr))

        elif (userInputStr == "valid"):
            print("Valid action-object combinations:")
            print(env.get_valid_action_object_combinations_with_templates())

        elif (userInputStr == 'goals'):
            print(env.get_goal_progress())

        else:
            # Send user input, get response
            observation, reward, isCompleted, info = env.step(userInputStr)
            if len(message)>1:
                message.append({"role": "user", "content": observation})
            score = info['score']
            print("【Observation】\n" + observation)
            # print("Reward: " + str(reward))
            print("Score: " + str(score), isCompleted)
            # print("isCompleted: " + str(isCompleted))
            # print("info: " + str(info))
            if score==100 or isCompleted==True:
                # 正常退出
                break

        # Select a random action
        valid_actions = env.get_valid_action_object_combinations()

        # Get user input
        if prompt_toolkit_available:
            actions_completer = WordCompleter(valid_actions, ignore_case=True, sentence=True)
            userInputStr = prompt('> ', completer=actions_completer,
                                  history=history, enable_history_search=True)
        else:
            userInputStr = get_response(message, model=model).replace("\n", "")
            print("【Action】", userInputStr)
            if "The task is complete" in userInputStr:
                break
            message.append({"role": "assistant", "content": userInputStr})
            step+=1
            if step>=MAX_STEP:
                # 超过步数退出
                break
        # Sanitize input
        userInputStr = userInputStr.lower().strip()

    # Display run history
    runHistory = env.get_run_history()
    for item in runHistory:
        print(item)
        print("")
    new_dict={"traj_id": taskName+"_"+str(args['var_num']), "instructions": TaskDescription, "messages": message, "is_completed": isCompleted, "score": score, "step": step}
    f=args['file']
    f.write(json.dumps(new_dict)+"\n")


def build_simplification_str(args):
    """ Build simplification_str from args. """
    simplifications = list()
    if args["teleport"]:
        simplifications.append("teleportAction")

    if args["self_watering_plants"]:
        simplifications.append("selfWateringFlowerPots")

    if args["open_containers"]:
        simplifications.append("openContainers")

    if args["open_doors"]:
        simplifications.append("openDoors")

    if args["no_electrical"]:
        simplifications.append("noElectricalAction")

    return args["simplifications_preset"] or ",".join(simplifications)

#   Parse command line arguments


def parse_args():
    desc = "Play through a game using the console."
    parser = argparse.ArgumentParser(desc)
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini", help="Model to use")
    parser.add_argument("--output_file", type=str, default="test_ours_4omini_bufferseq.jsonl", help="Output file path")
    parser.add_argument("--jar_path", type=str,
                        help="Path to the ScienceWorld jar file. Default: use builtin.")
    parser.add_argument("--task-num", type=int, default=13,
                        help="Specify the task number to play. Default: %(default)s")
    parser.add_argument("--var-num", type=int, default=0,
                        help="Specify the task variation number to play. Default: %(default)s")
    parser.add_argument("--env-step-limit", type=int, default=100,
                        help="Maximum number of steps per episode. Default: %(default)s")
    parser.add_argument("--num-episodes", type=int, default=5,
                        help="Number of episodes to play. Default: %(default)s")

    simplification_group = parser.add_argument_group('Game simplifications')
    simplification_group.add_argument(
        "--simplifications-preset", choices=['easy'],
        help="Choose a preset among:\n"
             "'easy' (teleportAction,openDoors,selfWateringFlowerPots,noElectricalAction)."
    )
    simplification_group.add_argument("--teleport", action="store_true",
                                      help="Lets agents instantly move to any location.")
    simplification_group.add_argument("--self-watering-plants", action="store_true",
                                      help="Plants do not have to be frequently watered.")
    simplification_group.add_argument("--open-containers", action="store_true",
                                      help="All containers are opened by default.")
    simplification_group.add_argument("--open-doors", action="store_true",
                                      help="All doors are opened by default.")
    simplification_group.add_argument("--no-electrical", action="store_true",
                                      help="Remove the electrical actions (reduces the size of the action space).")

    args = parser.parse_args()
    params = vars(args)
    return params


def main():
    print("ScienceWorld 1.0 API Examples - Human")
    id_list=[]
    args = parse_args()
    output_file=args['output_file']
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                id_list.append(json.loads(line)['traj_id'])
    f_write=open(output_file, 'a')
    args['file'] = f_write
    for i in open("./testset.jsonl", 'r'):
        task_id=json.loads(i)['traj_id']
        task_type=json.loads(i)['task_type']
        if task_id in id_list:
            continue
        id=task_id.find('_')
        str_id=task_id[id+1:]
        args['task_name']=task_type
        args['var_num']=int(str_id)
        args["simplification_str"] = build_simplification_str(args)
        userConsole(args)
    f_write.close()

if __name__ == "__main__":
    main()


# python human.py --model_name "qwen3-32b" --output_file "test_qwen3-32b_skill_alfsci.jsonl"
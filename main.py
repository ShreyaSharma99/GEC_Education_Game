from logging import exception
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random
from gym import spaces
# import json
import os
import copy
import math

from gec_env_class import *
from dataset import *

valid_commands = {"add", "replace", "delete", "hint", "undo", "index_hint", "all_hint", "idk", "giveup", "help", "quit", "next"}

if __name__ == '__main__':
    file_path_10gec = "10gec_annotations/"
    # file = open(file_path_10gec + "A1.m2")

    gec_data = Dataset(num_annotators = 10)
    gec_data.load_dataset(file_path_10gec)
    # print(gec_data.datapoints[74][0].bucket_change)
    # get_errors('Privicy protection belongs to human rights .', 'Privacy protection is a human right .')

    env = GEC_Env(gec_data)
    
    # print(env)
    user_name = input("Input your username : ")

    start_state = env.reset(user_name, 0)
    print("Your start state is :\n", start_state)
    print_sent(start_state)

    game_state_list = []
    task_count = 0

    while True:

        try:
            print("\n------------------------------------")
            user_input = input("\nInput Action: ")

            if "(" not in user_input: command = user_input.lower()
            else:                     command = user_input[:user_input.index("(")].lower()
            
            if command not in valid_commands:
                print("Invalid command!")
                continue 

            if command == "quit":
                comment = input("\nComments: ")
                game_state = env.save_game_state()
                game_state["comment"] = comment
                game_state["action_cmd"] = user_input
                game_state_list.append(game_state)
                json_object = json.dumps(game_state_list, indent = 4)
                with open(user_name + "_" + str(task_count) +  ".json", "w") as outfile:
                    outfile.write(json_object + "\n")
                break

            if command == "next":
                comment = input("\nComments: ")
                game_state = env.save_game_state()
                game_state["comment"] = comment
                game_state["action_cmd"] = user_input
                game_state_list.append(game_state)

                json_object = json.dumps(game_state_list, indent = 4)
                with open(user_name + "_" + str(task_count) +  ".json", "w") as outfile:
                    outfile.write(json_object + "\n")

                # reset everything
                task_count += 1
                if task_count > 5:
                    break
                game_state_list = []
                start_state = env.reset(user_name, task_count)
                print("\n-------------------- NEXT TASK --------------------\n")
                print("Your start state is :\n", start_state)
                print_sent(start_state)
                continue

            if user_input == "help":
                print("-----------------------\nChoose your action - [Add, Replace, Delete, Undo, Hint, Index_Hint, All_Hint, Idk, GiveUp, Done] \n" + 
                    "To move to the next correction task type - Next (if you reached the right answer) or GiveUp \n" +
                    "Action template: \n" + 
                    "Add([ind1, ind1], \"PHRASE YOU WANT TO ADD\") \n" + 
                    "Replace([ind1, ind2], \"PHRASE YOU WANT TO CHANGE TO\") \n" + 
                    "Delete([ind1, ind2]) \n" + 
                    "Undo \n"  +
                    "Hint - Get some hint about the next action with least penalty.\n" + 
                    "Index_Hint - Get some hint about \"where\" the next error is but with a slightly higher penalty.\n" + 
                    "All_Hint - Get some hint about \"where\" and \"what\" the next error is with a significant penalty.\n" + 
                    "Idk - Get one error corrected with high penalty.\n" + 
                    "GiveUp - It gives you the grammatically correct sentence itself and makes your reward to lowest.\n" +
                    "Next - When you reach one of the right answers and want to move to the next task.\n" + 
                    "\nIf you need help type \"help\" \nIf you want to quit type \"quit\"\n-----------------------")
                continue

            ind1, ind2 = -1, -1
            # if command not in env.action_space:
            #     raise exception("This action does not exist!\nTry executing one of these actions - Replace, Delete, Add, Undo, Hint, Index_Hint, All_Hint, Idk, GiveUp")

            modifier_action = ["add", "replace", "delete"]
            if command in modifier_action:
                index = user_input[user_input.index("[")+1 : user_input.index("]")]
                ind1, ind2 = int(index.split(",")[0]), int(index.split(",")[1])
            
            in_string = ""
            if "\"" in user_input:
                in_string = user_input[user_input.index("\"")+1 : -2]
            act1 = env.step(command, [ind1,ind2], in_string)
            print("\n")
            
            comment = input("\nComments: ")
            game_state = env.save_game_state()
            game_state["comment"] = comment
            game_state["action_cmd"] = user_input
            
            game_state_list.append(game_state)
        except:
            print("Invalid Input!!")
        # step += 1

    # Writing to sample.json
    # json_object = json.dumps(game_state_list, indent = 4)
    # with open(user_name + "_" + str(task_count) +  ".json", "a") as outfile:
    #     outfile.write(json_object + "\n")
    #     # break



# Replace([2,6], "is a")
# Replace([5,6], "right")
# Replace([0,1], "Privacy")

# Add([1,1], "the")
# Replace([7,8], "wanted")
'''
Hint
Index_Hint
All_Hint
Idk
Replace([7,8], "wanted")
Idk
Replace([8,9], "in")
All_Hint
Idk
GiveUp
'''


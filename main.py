from logging import exception
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random
from gym import spaces
import os
import copy
import math

from gec_env_class import *
from dataset import *

if __name__ == '__main__':
    file_path_10gec = "10gec_annotations/"
    file = open(file_path_10gec + "A1.m2")

    gec_data = Dataset(num_annotators = 10)
    gec_data.load_dataset(file_path_10gec)
    # print(gec_data.datapoints[74][0].bucket_change)
    # get_errors('Privicy protection belongs to human rights .', 'Privacy protection is a human right .')

    env = GEC_Env(gec_data)
    # print(env)
    start_state = env.reset()
    print("Your start state is :\n", start_state)
    print_sent(start_state)

    while True:
        # Add([1,1], "And then")
        # Replace([1,2], "He")
        # Delete([1,5])
        # Undo()
        print("\n") 
        user_input = input("\nInput Action: ")
        if user_input == "quit":
            break

        if user_input == "help":
            print("-----------------------\nChoose your action - [Add, Replace, Delete, Undo, Hint, Index_Hint, All_Hint, Idk, GiveUp] \n" + 
                "Action template: \n" + 
                "Add([ind1, ind1], \"PHRASE YOU WANT TO ADD\") \n" + 
                "Replace([ind1, ind2], \"PHRASE YOU WANT TO CHANGE TO\") \n" + 
                "Delete([ind1, ind2]) \n" + 
                "Undo \n"  +
                "Hint - Get some hint about the next action with least penalty.\n" + 
                "Index_Hint - Get some hint about \"where\" the next error is but with a slightly higher penalty.\n" + 
                "All_Hint - Get some hint about \"where\" and \"what\" the next error is with a significant penalty.\n" + 
                "Idk - Get one error corrected with high penalty.\n" + 
                "GiveUp - It gives you the grammatically correct sentence itself and makes your reward to 0.\n" + 
                "\nIf you need help type \"help\" \nIf you want to quit type \"quit\"\n-----------------------")
            continue
        # user_input = str(user_input)
        action = user_input.split("(")[0]
        ind1, ind2 = -1, -1

        if action not in env.action_space:
            raise exception("This action does not exist!\nTry executing one of these actions - Replace, Delete, Add, Undo, Hint, Index_Hint, All_Hint, Idk, GiveUp")

        modifier_action = ["Add", "Replace", "Delete"]
        if action in modifier_action:
            index = user_input[user_input.index("[")+1 : user_input.index("]")]
            ind1, ind2 = int(index.split(",")[0]), int(index.split(",")[1])
        
        in_string = ""
        if "\"" in user_input:
            in_string = user_input[user_input.index("\"")+1 : -2]
        act1 = env.step(action, [ind1,ind2], in_string)
        print("\n")
        if action == "GiveUp":
            break


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


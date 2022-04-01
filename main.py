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
    file_path_10gec = "../Datasets/10gec_annotations/"
    file = open(file_path_10gec + "A1.m2")

    gec_data = Dataset(num_annotators = 10)
    gec_data.load_dataset(file_path_10gec)

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
            print("-----------------------\nChoose your action - [Add, Replace, Delete, Undo] \nAction template: \nAdd([ind1, ind1], \"PHRASE YOU WANT TO ADD\") \nReplace([ind1, ind2], \"PHRASE YOU WANT TO CHANGE TO\") \nDelete([ind1, ind2]) \nUndo() \n \nIf you need help type \"help\" \nIf you want to quit type \"quit\"\n-----------------------")
            continue
        # user_input = str(user_input)
        action = user_input.split("(")[0]
        ind1, ind2 = -1, -1

        if not action == "Undo":
            index = user_input[user_input.index("[")+1 : user_input.index("]")]
            ind1, ind2 = int(index.split(",")[0]), int(index.split(",")[1])
        
        in_string = ""
        if "\"" in user_input:
            in_string = user_input[user_input.index("\"")+1 : -2]
        act1 = env.step(action, [ind1,ind2], in_string)
        print("\n")
    
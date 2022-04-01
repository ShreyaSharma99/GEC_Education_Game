from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random
from gym import spaces
import copy
from helper import *

class GEC_Env(Env):

    def __init__(self, dataset):

        self.sent = ""
        self.data_id = -1
        self.dataset = dataset
        self.data_state = ""
        self.gamma = 0.95
        self.annotator = 0
        self.feedback = ""
        
        self.reward = 0
        self.action_buffer = []
        self.sent_buff = []
        self.annotator_buff = []
        self.reward_buff = []
        # we create an observation space with predefined range
        self.observation_space = Box(low=np.array([len(dataset.vocab)] * dataset.max_sent_len, dtype=np.int32), high=np.array([0] * dataset.max_sent_len, dtype=np.int32), dtype = np.float32)
        # similar to observation, we define action space 
        self.action_space = ["Replace", "Delete", "Add", "Undo"]
        
    
    def step(self, action, arg1 = [], string = ""):
        sent1 = self.sent.split()
        if action == "Undo":
            done, info = True, {}
            if len(self.sent_buff) < 2:
                self.feedback = "Error : There is no action to undo!!"
                self.action_buffer.append([action, "", ""])
                print("Reward = ", self.reward)
                print(self.feedback, "\n")
                print("Next state: ", self.sent)
                print_sent(self.sent)
                return self.sent, self.reward, done, info 

            self.action_buffer.append([action, "", ""])
            # restoring previous state
            self.sent = self.sent_buff[-2]
            del self.sent_buff[-1]
            self.annotator =  self.annotator_buff[-2]
            del self.annotator_buff[-1]
            self.reward = self.reward_buff[-2]
            del self.reward_buff[-1]
            self.feedback = "Reverted back to old state"
            print("Reward = ", self.reward)
            print(self.feedback, "\n")
            print("Next state: ", self.sent)
            print_sent(self.sent)
            return self.sent, self.reward, done, info 

        phrase1, phrase2 = " ".join(self.sent.split()[arg1[0]:arg1[1]]), string
        # print("act: ", phrase1, phrase2)

        self.action_buffer.append([action, phrase1, phrase2])

        if action == "Delete":
            self.sent = " ".join(sent1[:arg1[0]] + sent1[arg1[1]:])
        
        elif action == "Add":
            self.sent = " ".join(sent1[:arg1[0]] + [string] + sent1[arg1[1]:])
            
        elif action == "Replace":
            self.sent = " ".join(sent1[:arg1[0]] + [string] + sent1[arg1[1]:])
        
        # new_reward = get_reward(self.sent, self.data_id, dataset)
        # if new_reward < self.reward:
        max_reward = 0
        for i in range(len(self.data_state)):
            annot = self.data_state[i]
            curr_reward = get_ied(self.sent.split(), annot.G.split())
            if max_reward < curr_reward:
                max_reward = curr_reward
                self.annotator = i

        action_seq = self.data_state[self.annotator].A_list
        self.feedback == ""
        if max_reward == 1:
            self.feedback = "Feedback: The sentence is grammatically correct! "
        else:
            if max_reward > self.reward:
                self.feedback = "Feedback: Yes, we are getting closer to the correct sentence! "
            else:
                self.feedback = "Feedback: You might want to recheck your last action! "
            for i in range(len(action_seq)):
                act = action_seq[i]
                p1 = " ".join(self.data_state[self.annotator].S.split()[act["indices"][0] : act["indices"][1]])
                p2 = self.data_state[self.annotator].A_list[i]["correction"]
            
                # # latest performed action matched 
                # if p1 == phrase1 and p2 == phrase2:

                # found action not performed yet
                if not in_buffer(self.action_buffer, p1, p2):
                    err = self.data_state[self.annotator].A_list[i]["error_tag"]
                    # TO DO: add level of feedback based on failed attempts
                    if err in FEEDBACK_TEMPLATE:
                        self.feedback += "Hint: There is an error in the " + FEEDBACK_TEMPLATE[err]
                    else:
                        self.feedback += "Hint: Action doesn't seem to be correct."
                    break
            # all actions done but still incorrect
            if "Hint" not in self.feedback:
                self.feedback += "Hint: You have done some unnecessary changes in the sentence. Undo the incorrect action(s)."

        # self.reward = math.pow(self.gamma, len(self.action_buffer)-1)*max_reward
        self.reward = max_reward

        print("Reward = ", self.reward)
        print(self.feedback, "\n")
        print("Next state: ", self.sent)
        print_sent(self.sent)
        done = True                            #Condition for completion of episode
        info = {}        

        self.sent_buff.append(self.sent)
        self.annotator_buff.append(self.annotator)
        self.reward_buff.append(self.reward)

        return self.sent, self.reward, done, info 

    def reset(self):
        #self.data_id = random.randint(0, data_size)-1
        self.data_id = 1262
        # 604
        print("data id = ",self.data_id)
        print("reward = ", self.reward)
        self.sent = copy.deepcopy(self.dataset.datapoints[self.data_id][0].S)
        self.data_state = copy.deepcopy(self.dataset.datapoints[self.data_id])
        self.action_buffer = []
        self.sent_buff = [self.sent]
        self.annotator_buff = [0]
        self.reward_buff = [0]
        # self.reward_unit = 1/max(len(dataset[self.data_id ][0]["A_list"]), 1)
        return self.sent  

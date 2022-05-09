from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random
from gym import spaces
import copy
import json

from helper import *

class GEC_Env(Env):

    def __init__(self, dataset):

        self.sent = ""
        # self.sent_mapping = []
        self.data_id = -1
        self.dataset = dataset
        self.data_state = ""
        self.gamma = 0.95

        self.annotator = 0
        self.edit_dist = dataset.max_sent_len
        self.reward = 0

        self.feedback = ""
        self.action_buff = []
        # self.reward = 0
        self.reward_buff = []
        
        self.sent_buff = []
        self.annotator_buff = []
        self.next_gt_action_buff = []
        self.edit_dist_buff = []

        # we create an observation space with predefined range
        self.observation_space = Box(low=np.array([len(dataset.vocab)] * dataset.max_sent_len, dtype=np.int32), high=np.array([0] * dataset.max_sent_len, dtype=np.int32), dtype = np.float32)
        # similar to observation, we define action space 
        self.action_space = ["replace", "delete", "add", "undo", "idk", "hint", "giveup", "index_hint", "all_hint"]
      
    
    def step(self, action, arg1 = [], string = ""):
        sent1 = self.sent.split()
        print("ACTION - ", action)
        if action == "undo":
            return self.action_undo()

        elif action == "hint":
            return self.action_hint(0)

        elif action == "idk":
            return self.action_idk()

        elif action == "giveup":
            return self.action_give_up()

        elif action == "index_hint":
            return self.action_hint(1)
        
        elif action == "all_hint":
            return self.action_hint(2)

        # update action buffer
        phrase1, phrase2 = " ".join(self.sent.split()[arg1[0]:arg1[1]]), string
        self.action_buff.append([action, phrase1, phrase2])

        if action == "delete":
            self.sent = " ".join(sent1[:arg1[0]] + sent1[arg1[1]:])    

        if action == "add" or action == "replace":
            self.sent = " ".join(sent1[:arg1[0]] + [string] + sent1[arg1[1]:])

        # calculate the maximum reward along with the annotator closest to the updated state
        min_edit = max(len(self.sent.split()), len(self.data_state[self.annotator].S.split()))
        for i in range(len(self.data_state)):
            annot = self.data_state[i]
            edit_dist = get_ed(self.sent.split(), annot.G.split())
            if min_edit > edit_dist:
                min_edit = edit_dist
                self.annotator = i

        pending_actions, action_len = get_errors(self.sent, self.data_state[self.annotator].G)
        self.feedback == ""
        if action_len == 0:
            self.feedback = "Feedback: The sentence is grammatically correct! "
            self.next_gt_action_buff.append(None)
        else:
            if min_edit < self.edit_dist:
                self.feedback = "Feedback: Yes, we are getting closer to the correct sentence! "
            else:
                self.feedback = "Feedback: You might want to recheck your last action! "

            # next_action_ind = pending_actions
            if len(pending_actions) == 0: # no action left to do
                # print("No action left to do!")
                # self.feedback += "Hint: You have done some unnecessary changes in the sentence. Undo the incorrect action(s)."
                self.next_gt_action_buff.append(None)
            else:
                # print("Next action suggested - ", pending_actions[0])
                err = pending_actions[0]["error_tag"]
                self.next_gt_action_buff.append(pending_actions[0])
                # TO DO: add level of feedback based on failed attempts
                phrase_feedback = get_phrase(self.next_gt_action_buff[-1])
                # if err in FEEDBACK_TEMPLATE1:
                #     self.feedback += "Hint: There is something "+ phrase_feedback +" with - " + FEEDBACK_TEMPLATE1[err][0]
                # else:
                #     self.feedback += "Hint: There is an something"+ phrase_feedback +" with - " + err

        self.reward = (self.edit_dist-min_edit)/len(self.data_state[self.annotator].S.split())
        self.edit_dist = min_edit
        print("Edit distance = ", self.edit_dist)
        print("Reward = ", self.reward)
        print("Closest Annotator = ", self.annotator)
        print(self.feedback, "\n")
        print("Next state: ", self.sent)
        print_sent(self.sent)
        done = True                            
        info = {}        

        self.sent_buff.append(self.sent)
        self.annotator_buff.append(self.annotator)
        self.reward_buff.append(self.reward)
        self.edit_dist_buff.append(self.edit_dist)

        return self.sent, self.reward, done, info 

    def action_undo(self):
        done, info = True, {}
        self.action_buff.append(["Undo", "", ""])
        if len(self.sent_buff) < 2:
            self.feedback = "Error : There is no action to undo!!"
            # self.reward = 0 - 0.25/len(self.data_state[self.annotator].S.split()) # undo penalty
            self.reward = 0.25/len(self.data_state[self.annotator].S.split()) # undo penalty
            self.reward_buff.append(self.reward)
            print("Reward = ", self.reward)
            print(self.feedback, "\n")
            print("Next state: ", self.sent)
            print_sent(self.sent)
            return self.sent, self.reward, done, info 
        
        # self.reward = self.reward_buff[-2] - 0.25/len(self.data_state[self.annotator].S.split()) # undo penalty
        self.reward = -0.25/len(self.data_state[self.annotator].S.split()) # undo penalty
        self.reward_buff.append(self.reward)
        
        # restoring previous state
        self.sent = self.sent_buff[-2]
        del self.sent_buff[-1]
        self.annotator =  self.annotator_buff[-2]
        del self.annotator_buff[-1]
        self.edit_dist = self.edit_dist_buff[-2]
        del self.edit_dist_buff[-1]
        del self.next_gt_action_buff[-1]

        self.feedback = "Reverted back to old state"
        print("Reward = ", self.reward)
        print(self.feedback, "\n")
        print("Next state: ", self.sent)
        print_sent(self.sent)
        return self.sent, self.reward, done, info 

    def action_hint(self, hint_type):
        if len(self.next_gt_action_buff) == 0:
            raise Exception("ERROR! Empty next_gt_action_buff")

        if self.next_gt_action_buff[-1] == None:
            self.feedback = "Hint: The sentence is already grammatically correct!"
        else:
            # print(self.next_gt_action_buff)
            err_tag = self.next_gt_action_buff[-1]["error_tag"]
            phrase_feedback = get_phrase(self.next_gt_action_buff[-1])
            example_feedback = FEEDBACK_TEMPLATE1[err_tag][0] + "\n" + FEEDBACK_TEMPLATE1[err_tag][1] if err_tag in FEEDBACK_TEMPLATE1 else "Feedback: " + err_tag
            if hint_type == 0:  # example type feedback
                self.feedback = "Hint: There is something " + phrase_feedback + " with - " + example_feedback
                self.reward = -0.5/len(self.data_state[self.annotator].S.split()) # hint penalty for type 0 
            elif hint_type == 1:  # index type feedback
                self.feedback = "Hint: There is something " + phrase_feedback + " between indices - " + str(self.next_gt_action_buff[-1]["indices"][0]) + " and " + str(self.next_gt_action_buff[-1]["indices"][1])
                self.reward = -0.6/len(self.data_state[self.annotator].S.split()) # hint penalty for type 1 
            elif hint_type == 2:  # index + error type feedback
                self.feedback = "Hint: There is something " + phrase_feedback + " between indices - " + str(self.next_gt_action_buff[-1]["indices"][0]) + " and " + str(self.next_gt_action_buff[-1]["indices"][1]) + \
                                " with error type - " + example_feedback
                self.reward = -0.8/len(self.data_state[self.annotator].S.split()) # hint penalty for type 2
                                
        
        self.action_buff.append(["Hint", "", ""])
        self.reward_buff.append(self.reward)
        print("Reward = ", self.reward)
        print(self.feedback, "\n")
        print("Next state: ", self.sent)
        print_sent(self.sent)
        return self.sent, self.reward, True, {} 

    
    def action_give_up(self):
        print("inside giveup")
        self.feedback = "Feedback: One of the grammatically correct answers is - \n" +  self.data_state[self.annotator].G + "\n"
        self.reward = -100
        self.reward_buff.append(self.reward)
        self.sent = self.data_state[self.annotator].G
        self.sent_buff.append(self.sent)
        self.next_gt_action_buff.append(None)
        self.edit_dist = 0
        self.edit_dist_buff.append(self.edit_dist)
        self.action_buff.append(["GiveUp", "", ""])
        print("Reward = ", self.reward)
        print(self.feedback, "\n")
        print("Next state: ", self.sent)
        print_sent(self.sent)
        return self.sent, self.reward, True, {} 

    def action_idk(self):
        if self.next_gt_action_buff[-1] == None:
            self.feedback = "Hint : The sentence is already grammatically correct!"
        else:
            error = self.next_gt_action_buff[-1]["error_tag"] if self.next_gt_action_buff[-1]["error_tag"] not in FEEDBACK_TEMPLATE1 else FEEDBACK_TEMPLATE1[self.next_gt_action_buff[-1]["error_tag"]][0]
            self.feedback = "Major Hint: One of the grammatical errors - " +  error + ", have been corrected!"
            self.sent = perform_act(self.sent, self.next_gt_action_buff[-1])
        # self.reward = self.reward - 1/len(self.data_state[self.annotator].S.split()) # idk penalty
        self.reward = - 1/len(self.data_state[self.annotator].S.split()) # idk penalty
        self.reward_buff.append(self.reward)
        self.sent_buff.append(self.sent)
        pending_act, _ = get_errors(self.sent, self.data_state[self.annotator].G)
        if len(pending_act)>0: self.next_gt_action_buff.append(pending_act[0])
        else: self.next_gt_action_buff.append(None)
        self.edit_dist = len(pending_act)
        self.edit_dist_buff.append(self.edit_dist)
        self.action_buff.append(["Idk", "", ""])
        print("Reward = ", self.reward)
        print(self.feedback, "\n")
        print("Next state: ", self.sent)
        print_sent(self.sent)
        return self.sent, self.reward, True, {} 


    def reset(self, user, count):
        #self.data_id = random.randint(0, data_size)-1
        task_ind = {'mattia' : [3, 635, 35, 9, 12, 52],
                    'shehzaad' : [106, 159, 112, 209, 635, 3]}
        self.data_id = task_ind[user][count]
        # 604, 1262
        print("data id = ",self.data_id)
        print("reward = ", self.reward)
        self.sent = copy.deepcopy(self.dataset.datapoints[self.data_id][0].S)
        self.data_state = copy.deepcopy(self.dataset.datapoints[self.data_id])
        # self.sent_mapping = [list(np.arange(0, len(self.sent.split())))]
        self.annotator = 0
        self.feedback = ""
        self.reward = 0
        self.bucket_map = {}
        for i in range(len(self.sent.split())):
            self.bucket_map[i] = 2*i + 1

        self.edit_dist = get_ed(self.sent.split(), self.data_state[self.annotator].G.split())
        # self.edit_dist = len(self.data_state[self.annotator].S.split())

        self.reward_buff = [0]
        self.edit_dist_buff = [self.edit_dist]
        self.action_buff = []
        self.sent_buff = [self.sent]
        self.annotator_buff = [0]
        self.next_gt_action_buff = []
        pending_act, _ = get_errors(self.sent, self.data_state[self.annotator].G)
        if len(pending_act)>0:
            self.next_gt_action_buff = [pending_act[0]]
        # self.reward_unit = 1/max(len(dataset[self.data_id ][0]["A_list"]), 1)
        return self.sent  

    def toJson(self):
        return json.dumps(self, default=lambda o: o.__dict__)

    def save_game_state(self):
        state_dict = {}
        state_dict["sent"] = self.sent
        state_dict["data_id"] = self.data_id
        # state_dict["data_state"] = self.data_state
        state_dict["gamma"] = self.gamma
        state_dict["annotator"] = self.annotator
        state_dict["edit_dist"] = self.edit_dist
        state_dict["reward"] = self.reward
        state_dict["feedback"] = self.feedback
        state_dict["action_buff"] = self.action_buff
        state_dict["reward_buff"] = self.reward_buff
        state_dict["sent_buff"] = self.sent_buff
        state_dict["annotator_buff"] = self.annotator_buff
        state_dict["next_gt_action_buff"] = self.next_gt_action_buff
        state_dict["edit_dist_buff"] =self.edit_dist_buff

        return state_dict

        # self.sent = ""
        # # self.sent_mapping = []
        # self.data_id = -1
        # self.dataset = dataset
        # self.data_state = ""
        # self.gamma = 0.95

        # self.annotator = 0
        # self.edit_dist = dataset.max_sent_len
        # self.reward = 0

        # self.feedback = ""
        # self.action_buff = []
        # self.reward = 0
        # self.reward_buff = []
        
        # self.sent_buff = []
        # self.annotator_buff = []
        # self.next_gt_action_buff = []
        # self.edit_dist_buff = []
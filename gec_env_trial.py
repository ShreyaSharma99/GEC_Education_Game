
# %%
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random
from gym import spaces
import os
import copy
import math


# %%

def print_sent(sent):
    table = []
    sent = sent.split()
    for i in range(len(sent)):
        print(sent[i]+"(" + str(i) + ")", end = " ")
        
# print_sent("In the old days , if one wants to tell some important news to another one , which lives far away , he needs to write letters and it wastes time .")


# %%
# inpute one annotator correction in the fporm of a dictionary like -
# {'S': 'Genetic risk refers more to your chance of inheriting a disorder or disease .', 
# 'A_list': [{'indices': ['3', '4'], 'error_tag': 'Rloc-', 'correction': ''}]}
def get_correct(data):
    sent = data["S"].split()
    shift = 0
    for action in data["A_list"]:
        index = action["indices"]
        # print_sent(" ".join(sent))
        # print(shift)
        # print(index[0]+shift, index[1]+shift)
        # print(sent)
        # print(sent[index[0]+shift], sent[index[1]+shift])
        sent = sent[:index[0]+shift] + action["correction"].split() + sent[index[1]+shift:]
        shift += len(action["correction"].split()) - (index[1] - index[0])
        # if index[0] == index[1]: shift += 1  
    return " ".join(sent)


# %%
# string matching reward
def get_ied(instseq1, instseq2):
    m = len(instseq1)
    n = len(instseq2)
    if min(m,n) == 0: return 0
    cost_matrix = {}
    for i in range(m+1):
        for j in range(n+1):
            if min(i,j) == 0:
                cost_matrix[str(i)+'_'+str(j)] = max(i,j)
                continue
            cost = 1
            try:
                if instseq1[i-1] == (instseq2[j-1]):
                    cost = 0
            except:
                raise Exception('levenshtein calculation error')
            a = cost_matrix[str(i-1)+'_'+str(j)]+1
            b = cost_matrix[str(i)+'_'+str(j-1)]+1
            c = cost_matrix[str(i-1)+'_'+str(j-1)]+cost
            cost_matrix[str(i)+'_'+str(j)] = min(a, min(b,c))
    ed = float(cost_matrix[str(m)+'_'+str(n)])
    return 1 - (ed / max(m,n))

# %%
error_tags = ['ArtOrDet', 'Mec', 'Nn', 'Npos', 'Others', 'Pform', 'Pref', 'Prep', 'Rloc-', 'SVA', 'Sfrag', 'Smod', 
              'Srun', 'Ssub', 'Trans', 'Um', 'V0', 'Vform', 'Vm', 'Vt', 'WOadv', 'WOinc', 'Wci', 'Wform', 'Wtone']
feedback_template = {}
standard_string = "missing or incorrect"
feedback_template = {'ArtOrDet' : "Article", 
                  'Mec' : "Punctuation / capitalization / spelling",
                  'Nn' : "Noun number incorrect",
                  'Npos' : "Possesive Noun",
                  'Pform' : "Pronoun Form",
                  'Pref' : "Pronoun reference",
                  'Prep' : "Preposition",
                  'Rloc-' : "Local Redundency",
                  'SVA' : "Subject-verb-agreement",
                  'Sfrag' : "Sentence fragmant",
                  'Smod' : "Dangling Modifier",  #modifier could be misinterpreted as being associated with a word other than the one intended
                  'Srun' : "Runons / comma splice",  # fault is the use of a comma to join two independent clauses
                  'Ssub' : "Subordinate clause", # "I know that Bette is a dolphin" - here "that Bette is a dolphin" occurs as the complement of the verb "know" 
                  'Trans' : "Conjuctions",
                  'V0' : "Missing verb",
                  'Vform' : "Verb form",
                  'Vm' : "Verb modal",
                  'Vt' : "Verb tense",
                  'WOadv' : "Adverb/adjective position",
                  'Wci' : " Wrong collocation", # it went fair well -> fairly
                  'Wform' : "Word form"
                    }


# generating feedback with respect to annotator 1
def generate_feedback(datapoint):
    return feedback_template[datapoint[0]["error_tag"]]

# READ DATASET
# %%
file_path_10gec = "../Datasets/10gec_annotations/"
file = open(file_path_10gec + "A1.m2")

Sample_sent = []
Vocab = set()
error_tags = set()
max_len = 0

for line in file.readlines():
    if line[0] == "S" and len(line[2:].strip())>0:
        Sample_sent.append(line[2:].strip())
        Vocab.update(set(line[2:].strip().split()))
        max_len = max(max_len, len(line[2:].strip().split()))
    if line[0] == "A":
        words = line.split("|||")
        Vocab.update(set(words[2].split()))
        error_tags.update(set(words[1].split()))
Vocab = list(Vocab) + ["reflected"]
vocab_size = len(Vocab)


# %%
# for e in sorted(list(error_tags)):
#     print("'" + e + "'", end=", ")


# %%
# Extract ground truth correct sentences for grammatically incorect sentences
# Ex: S The opposite is also true .
#     A 5 6|||Mec|||:|||REQUIRED|||-NONE-|||0

dataset = {}
annotators = 10
for an in range(1, 11):
    file = "A" + str(an) + ".m2"
    file = open(file_path_10gec + file)
    current_S = ""
    current_A = []
    ex_ind = -1
    for line in file.readlines():
        if line[0] == "S":
            ex_ind += 1
            current_S = line[2:].strip()
        elif line[0] == "A":
            ans  = {}
            cells = line.strip().split("|||")
            ans["indices"] = [int(i) for i in cells[0].split()[1:]]  # A 5 6
            ans["error_tag"] = cells[1]
            ans["correction"] = cells[2]
            current_A.append(ans)
        elif line.strip() == "":  # empty line
            annot = {"S" : current_S, "A_list" : current_A}
            dataset[ex_ind] = [annot] if an==1 else dataset[ex_ind] + [annot]
            dataset[ex_ind][an-1]["G"] = get_correct(dataset[ex_ind][an-1])
            current_S = ""
            current_A = []
        
data_size = len(dataset.keys())

# %%
def get_reward(sent, data_id, dataset):
    data = dataset[data_id]
    max_reward = 0
    for annotator in dataset[data_id]:
        G = annotator["G"]
        max_reward = max(max_reward, get_ied(sent.split(), G.split()))
    return max_reward          
    
# %%
def in_buffer(action_buffer, p1, p2):
    for act in action_buffer:
        if act[1] == p1 and act[2] == p2:
            return True
    return False

# %%
class OurCustomEnv(Env):

    def __init__(self):

        self.sent = ""
        self.data_id = -1
        self.data_state = ""
        self.gamma = 0.95
        self.annotator = 0
        self.feedback = ""

        self.reward = 0
        self.action_buffer = []
        self.sent_buff = []
        self.annotator_buff = []
        self.reward_buff = []
        # self.high =np.array([vocab_size] * max_len, dtype=np.int32)
        # self.low =np.array([0] * max_len, dtype=np.int32)
        
        # self.reward_unit = 10e-10
        #we create an observation space with predefined range
        self.observation_space = Box(low=np.array([vocab_size] * max_len, dtype=np.int32), high=np.array([0] * max_len, dtype=np.int32), dtype = np.float32)

        #similar to observation, we define action space 
        self.action_space = ["Replace", "Delete", "Add", "Undo"]
        
    
    def step(self, action, arg1 = [], string = ""):
        sent1 = self.sent.split()
        if action == "Undo":
            done, info = True, {}
            if len(self.sent_buff) < 2:
                self.feedback = "Error : There is no action to undo!!"
                self.action_buffer.append([action, "", ""])
                print("Reward = ", self.reward)
                print(self.feedback)
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
            print(self.feedback)
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
            curr_reward = get_ied(self.sent.split(), annot["G"].split())
            if max_reward < curr_reward:
                max_reward = curr_reward
                self.annotator = i

        action_seq = self.data_state[self.annotator]["A_list"]
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
                p1 = " ".join(self.data_state[self.annotator]["S"].split()[act["indices"][0] : act["indices"][1]])
                p2 = self.data_state[self.annotator]["A_list"][i]["correction"]
            
                # # latest performed action matched 
                # if p1 == phrase1 and p2 == phrase2:

                # found action not performed yet
                if not in_buffer(self.action_buffer, p1, p2):
                    err = self.data_state[self.annotator]["A_list"][i]["error_tag"]
                    # TO DO: add level of feedback based on failed attempts
                    if err in feedback_template:
                        self.feedback += "Hint: There is an error in the " + feedback_template[err]
                    else:
                        self.feedback += "Hint: Action doesn't seem to be correct."
                    break
            # all actions done but still incorrect
            if "Hint" not in self.feedback:
                self.feedback += "Hint: You have done some unnecessary changes in the sentence. Undo the incorrect action(s)."

        # self.reward = math.pow(self.gamma, len(self.action_buffer)-1)*max_reward
        self.reward = max_reward

        print("Reward = ", self.reward)
        print(self.feedback)
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
        # print("reward = ", self.reward)
        self.sent = copy.deepcopy(dataset[self.data_id ][0]["S"])
        self.data_state = copy.deepcopy(dataset[self.data_id ])
        self.action_buffer = []
        self.sent_buff = [self.sent]
        self.annotator_buff = [0]
        self.reward_buff = [0]
        # self.reward_unit = 1/max(len(dataset[self.data_id ][0]["A_list"]), 1)
        return self.sent
    
def act(action, ind1=[], string=""):
    # if action == "Delete":
    #     return env.step("Delete", ind1)
    # else:
    return env.step(action, ind1, string)    


# %%
env = OurCustomEnv()


# %%
state = env.reset()
# state


# %%
print(state)
print_sent(state)

while True:
    # Add([1,1], "And then")
    # Replace([1,2], "He")
    # Delete([1,5])
    # Undo()
    user_input = input("\nInput Action: ")
    if user_input == "quit":
        break

    if user_input == "help":
        print("Choose your action - [Add, Replace, Delete, Undo] \nAction template: \nAdd([ind1, ind1], \"PHRASE YOU WANT TO ADD\") \nReplace([ind1, ind2], \"PHRASE YOU WANT TO CHANGE TO\") \nDelete([ind1, ind2]) \nUndo() \n \n If you need help type \"help\" \nIf you want to quite type \"quit\"")
        continue
    # user_input = str(user_input)
    action = user_input.split("(")[0]
    ind1, ind2 = -1, -1

    if not action == "Undo":
        index = user_input[user_input.index("[")+1 : user_input.index("]")]
        ind1, ind2 = int(index.split(",")[0]), int(index.split(",")[1])
    
    in_string = ""
    if "\"" in user_input:
        in_string = user_input[user_input.index("\"") : -2]
    act1 = act(action, [ind1,ind2], in_string)
    
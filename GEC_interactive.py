#!/usr/bin/env python
# coding: utf-8

# # 0. Install Dependencies

# In[1]:


# get_ipython().system('pip install tensorflow==2.3.0')
# get_ipython().system('pip install gym')
# get_ipython().system('pip install keras')
# get_ipython().system('pip install keras-rl2')


# # 1. Test Toy Environment with OpenAI Gym

# In[1]:


from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random
from gym import spaces
import os
import copy


# In[2]:


# inpute one annotator correction in the fporm of a dictionary like -
# {'S': 'Genetic risk refers more to your chance of inheriting a disorder or disease .', 
# 'A_list': [{'indices': ['3', '4'], 'error_tag': 'Rloc-', 'correction': ''}]}
def get_correct(data):
    sent = data["S"].split()
    shift = 0
    for action in data["A_list"]:
        index = action["indices"]
        sent = sent[:index[0]+shift] + [action["correction"]] + sent[index[1]+shift:]
        if index[0] == index[1]: shift += 1  
    return " ".join(sent)


# In[3]:


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

def print_sent(sent):
    table = []
    sent = sent.split()
    for i in range(len(sent)):
        print(sent[i]+"(" + str(i) + ")", end = " ")


# In[4]:


""" datapoint 301
[{'S': 'One of the diseases is sickle cell trait .',
  'A_list': [{'indices': [2, 3],
    'error_tag': 'ArtOrDet',
    'correction': 'these'}],
  'G': 'One of these diseases is sickle cell trait .'},
 {'S': 'One of the diseases is sickle cell trait .',
  'A_list': [{'indices': [1, 2], 'error_tag': 'Prep', 'correction': ''},
   {'indices': [2, 3], 'error_tag': 'ArtOrDet', 'correction': 'such'},
   {'indices': [3, 4], 'error_tag': 'Nn', 'correction': 'disease'}],
  'G': 'One  such disease is sickle cell trait .'}, ..........]
"""


# In[5]:


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


# In[6]:


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


# In[7]:


for e in sorted(list(error_tags)):
    print("'" + e + "'", end=", ")


# In[17]:


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


# In[9]:


# # dump the json in a file
# import json
# import os
# with open('./10gec_jsons/data.json', 'w') as fh:
#     json.dump(dataset, fh, indent=2)


# In[10]:


# vocab = ["", "I", "he", "she", "they", "am", "is", "are", "was", "were", "go", "going", "goes", "went", "market", 
#          "amusement-park", "to", "a", "an", "the"]
# vocab_size = len(vocab)
# max_len = 10
# Sample_sent = ["I is going to market.", "She am go to market.", "They are go to a amusement-park.", 
#                "He going to an market.", "I were going."]


# In[11]:


def get_reward(sent, data_id, dataset):
    data = dataset[data_id]
    max_reward = 0
    for annotator in dataset[data_id]:
        G = annotator["G"]
        max_reward = max(max_reward, get_ied(sent.split(), G.split()))
    return max_reward             
    


# In[12]:


s = 'People get certain diseases because of changes .'
get_reward(s, 3, dataset)


# In[13]:


class OurCustomEnv(Env):

    def __init__(self):

        self.sent = ""
        self.data_id = -1
        self.data_state = ""
        
        self.high =np.array([vocab_size] * max_len, dtype=np.int32)
        self.low =np.array([0] * max_len, dtype=np.int32)
        
        self.reward = 0
        self.reward_unit = 10e-10
        #we create an observation space with predefined range
        self.observation_space = Box(low=self.low, high=self.high, dtype = np.float32)

        #similar to observation, we define action space 
        self.action_space = ["Replace", "Delete", "Add"]
        
    
    def step(self, action, arg1, string = ""):
        sent1 = self.sent.split()
        phrase1, phrase2 = " ".join(self.sent.split()[arg1[0]:arg1[1]]), string
        print("act: ", phrase1, phrase2)
        
        if action == "Delete":
            # self.sent = " ".join(sent1[:arg1] + sent1[arg1+1:])
            self.sent = " ".join(sent1[:arg1[0]] + sent1[arg1[1]:])
        
        elif action == "Add":
            # self.sent = " ".join(sent1[:arg1] + [Vocab[arg2]] + sent1[arg1:])
            self.sent = " ".join(sent1[:arg1[0]] + [string] + sent1[arg1[1]:])
            
        elif action == "Replace":
            # self.sent = " ".join(sent1[:arg1] + [Vocab[arg2]] + sent1[arg1+1:])
            self.sent = " ".join(sent1[:arg1[0]] + [string] + sent1[arg1[1]:])
        
        new_reward = get_reward(self.sent, self.data_id, dataset)
        # if new_reward < self.reward:
        action_seq = self.data_state["A_list"]
        add_reward = False
        for i in range(len(action_seq)):
            act = action_seq[i]
            p1 = " ".join(self.data_state["S"].split()[act["indices"][0] : act["indices"][1]])
            p2 = self.data_state["A_list"][i]["correction"]
            print(i, p1, p2)
            if p1 == phrase1 and p2 == phrase2:
                add_reward = True
                del self.data_state["A_list"][i]
                break
            
        if add_reward:
            self.reward += self.reward_unit
        elif len(self.data_state["A_list"]) == 0:
            print("Feedback: The sentence was grammatically correct already! Undo changes")
        else:
            err = self.data_state["A_list"][0]["error_tag"]
            if err in feedback_template:
                print("Feedback: Action doesn't match. There is an error in ", feedback_template[err])
            else:
                print("Feedback: Action doesn't match.")
            self.reward = max(0, self.reward - (self.reward_unit/2))
        
            
        done = True                            #Condition for completion of episode
        info = {}        

        return self.sent, self.reward, done, info 

    def reset(self):
        #self.data_id = random.randint(0, data_size)-1
        self.data_id = 1262
        # 604
        print("data id = ",self.data_id)
        print("reward = ", self.reward)
        self.sent = copy.deepcopy(dataset[self.data_id ][0]["S"])
        self.data_state = copy.deepcopy(dataset[self.data_id ][0])
        # self.reward_unit = 1/max(len(dataset[self.data_id ][0]["A_list"]), 1)
        return self.sent
    
def act(action, ind1, string=""):
    if action == "Delete":
        return env.step("Delete", ind1)
    else:
        return env.step(action, ind1, string)    
    
#         elif action == "Add_conjuction":
#             sent1 = self.sent.split()
#             self.sent = " ".join(sent1[:arg1] + [Vocab[arg2]] + sent1[arg1+1:])
            
#         elif action == "Replace_conjuction":
#             sent1 = self.sent.split()
#             self.sent = " ".join(sent1[:arg1] + [Vocab[arg2]] + sent1[arg1+1:])
            
#         elif action == "Replace_preposition":
#             sent1 = self.sent.split()
#             self.sent = " ".join(sent1[:arg1] + [Vocab[arg2]] + sent1[arg1+1:])
#         reward = 0.0


# In[14]:


env = OurCustomEnv()


# In[15]:


state = env.reset()
state


# In[16]:


print_sent(state)


# In[67]:


act1 = act("Replace", [5,6], "someone")
print(act1[1])
print_sent(act1[0])


# In[68]:


act2 = act("Delete", [0,1], "")
print(act2[1])
print_sent(act2[0])


# In[69]:


act2 = act("Add", [0,1], "In")
print(act2[1])
print_sent(act2[0])


# In[23]:


# act3 = act("Add", 17, "one")
# act3


# In[ ]:


act4 = act("Add", 18, "of")
act4


# In[70]:


dataset[604]


# In[ ]:





# In[ ]:





from re import I
from constants import *
import errant

# {'indices': [5, 6], 'error_tag': 'Pform', 'correction': 'someone'},
def get_errors(sent, gt):
    annotator = errant.load('en')
    org = annotator.parse(sent)
    corr = annotator.parse(gt)
    edits = annotator.annotate(org, corr)
    actions = []
    for e in edits:
        act = {"indices" : [e.o_start, e.o_end], "error_tag": e.type[2:], "correction": e.c_str}
        actions.append(act)
        # print(e.o_start, e.o_end, e.o_str, e.c_start, e.c_end, e.c_str, e.type)
    return actions, len(edits)

def print_sent(sent):
    table = []
    sent = sent.split()
    for i in range(len(sent)):
        print(sent[i]+"(" + str(i) + ")", end = " ")

def get_correct(annot):
    # data is an annotator object
    sent = annot.S.split()
    shift = 0
    for action in annot.A_list:
        index = action["indices"]
        sent = sent[:index[0]+shift] + action["correction"].split() + sent[index[1]+shift:]
        shift += len(action["correction"].split()) - (index[1] - index[0])
        # if index[0] == index[1]: shift += 1  
    return " ".join(sent)

def get_ed(instseq1, instseq2):
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
    # return 1 - (ed / max(m,n))
    return ed

# generating feedback with respect to annotator 1
def generate_feedback(datapoint):
    return FEEDBACK_TEMPLATE1[datapoint[0]["error_tag"]][0]

# def get_reward(sent, data_id, dataset):
#     data = dataset[data_id]
#     max_reward = 0
#     for annotator in dataset[data_id]:
#         G = annotator["G"]
#         max_reward = max(max_reward, get_ed(sent.split(), G.split()))
#     return max_reward       

def in_buffer(action_buffer, p1, p2):
    for act in action_buffer:
        if act[1] == p1 and act[2] == p2:
            return True
    return False

def perform_act(sent, action):
    # act = {'indices': [5, 6], 'error_tag': 'Pform', 'correction': 'someone'}
    sent = sent.split()
    index = action["indices"]
    sent_new = sent[:index[0]] + action["correction"].split() + sent[index[1]:]
    return " ".join(sent_new)

# def find_undone_action(annot, next_S):
#     next_S_list = next_S.split()
#     sent = annot.S.split()
#     act_list = annot.A_list
#     # For add action - if a in act_list when acted on S gives S' not < of state then suggest this action or replace action
#     # Get index wrt to original S for the current action agent performs and match Delete action
    
#     for i in range(len(act_list)):
#         if act_list[i]["correction"] == "":  # delete action
#             ind = act_list[i]['indices']
#             delete_phrase = " ".join(sent[ind[0]:ind[1]])
#             if delete_phrase in next_S:
#                 return i
#         else:  # replace / add action
#             s_new = perform_act(sent, act_list[i])
#             count = 0
#             # s_new should be  sub-sequence of next_s
#             for s in s_new:
#                 while count < len(next_S) and s!=next_S[count]:
#                     count += 1
#                 if count == len(next_S): # not a sub-sequence
#                     return i
#                 # s==next_S[count]
#                 count += 1

#     return -1

def get_phrase(action):
    phrase_feedback = "incorrect"
    if action["indices"][0] == action["indices"][1]: # add
        phrase_feedback = "missing"
    elif action["correction"] == "": # delete
        phrase_feedback = "extra"
    return phrase_feedback

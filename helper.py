from constants import *

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
        # print_sent(" ".join(sent))
        # print(shift)
        # print(index[0]+shift, index[1]+shift)
        # print(sent)
        # print(sent[index[0]+shift], sent[index[1]+shift])
        sent = sent[:index[0]+shift] + action["correction"].split() + sent[index[1]+shift:]
        shift += len(action["correction"].split()) - (index[1] - index[0])
        # if index[0] == index[1]: shift += 1  
    return " ".join(sent)

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

# generating feedback with respect to annotator 1
def generate_feedback(datapoint):
    return FEEDBACK_TEMPLATE[datapoint[0]["error_tag"]]


def get_reward(sent, data_id, dataset):
    data = dataset[data_id]
    max_reward = 0
    for annotator in dataset[data_id]:
        G = annotator["G"]
        max_reward = max(max_reward, get_ied(sent.split(), G.split()))
    return max_reward       

def in_buffer(action_buffer, p1, p2):
    for act in action_buffer:
        if act[1] == p1 and act[2] == p2:
            return True
    return False
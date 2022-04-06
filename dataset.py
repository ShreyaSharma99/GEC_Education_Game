from constants import *
from helper import *

# Sample Datapoint - 
# {'S': 'In old days , if one wants to tell some important news to another one , which lives far away , he needs to write letters and it wastes time .',
# 'A_list': [{'indices': [1, 1], 'error_tag': 'ArtOrDet', 'correction': 'the'},
# {'indices': [5, 6], 'error_tag': 'Pform', 'correction': 'someone'},
# {'indices': [6, 7], 'error_tag': 'Vt', 'correction': 'wanted'},
# {'indices': [13, 16], 'error_tag': 'Wci', 'correction': 'someone else'},
# {'indices': [16, 17], 'error_tag': 'Pform', 'correction': 'who'},
# {'indices': [17, 18], 'error_tag': 'Vt', 'correction': 'lived'},
# {'indices': [22, 23], 'error_tag': 'Vt', 'correction': 'needed'},
# {'indices': [26, 28], 'error_tag': 'Ssub', 'correction': 'which'}],
# 'G': 'In the old days , if someone wanted to tell some important news to someone else who lived far away , he needed to write letters which wastes time .'},

class Annotator():
    def __init__(self, S, A_list=[], G=""):
        self.S = S
        self.A_list = A_list
        self.G = G

class Dataset():
    def __init__(self, num_annotators = 10):
        self.num_annotators = num_annotators
        self.datapoints = []
        self.error_tags = set()
        self.vocab = set()
        self.max_sent_len = 0

    def load_dataset(self, file_name):
        for an in range(1, self.num_annotators+1):
            file = "A" + str(an) + ".m2"
            file = open(file_name + file)
            current_S = ""
            current_A = []
            ex_ind = -1
            for line in file.readlines():
                if line[0] == "S":
                    ex_ind += 1
                    current_S = line[2:].strip()
                    self.vocab.update(set(current_S.split()))
                    self.max_sent_len = max(self.max_sent_len, len(current_S.split()))

                elif line[0] == "A":
                    ans  = {}
                    cells = line.strip().split("|||")
                    ans["error_tag"] = cells[1]
                    if ans["error_tag"] == "Um":
                        continue
                    ans["indices"] = [int(i) for i in cells[0].split()[1:]]  # A 5 6
                    ans["correction"] = cells[2]
                    current_A.append(ans)

                    # if ans["indices"][1] == ans["indices"][0]: # "add" action
                    #     current_A_bucket.append({"bucket" : ans["indices"][1]*2, "correction": ans["correction"], "error_tag":ans["error_tag"]})

                    # elif ans["indices"][1] > ans["indices"][0] and ans["correction"]=="":  # "delete" action
                    #     for i in range(ans["indices"][0], ans["indices"][1]):
                    #         current_A_bucket.append({"bucket" : i*2 + 1, "correction": "", "error_tag":ans["error_tag"]})

                    # elif ans["indices"][1] > ans["indices"][0] and ans["correction"]!="":  # "replace" action
                    #     current_A_bucket.append({"bucket" : ans["indices"][0]*2, "correction": ans["correction"], "error_tag":ans["error_tag"]})
                    #     for i in range(ans["indices"][0], ans["indices"][1]):
                    #         current_A_bucket.append({"bucket" : i*2 + 1, "correction": "", "error_tag":ans["error_tag"]})

                    self.vocab.update(set(ans["correction"].split()))
                    self.error_tags.update(set(ans["error_tag"]))

                elif line.strip() == "" and ex_ind  >=0:  # empty line
                    annot = Annotator(current_S, current_A)
                    # {"S" : current_S, "A_list" : current_A}
                    if an==1:
                        an_list = [annot]
                        self.datapoints.append(an_list)
                    else:
                        self.datapoints[ex_ind].append(annot)
                    self.datapoints[ex_ind][an-1].G = get_correct(self.datapoints[ex_ind][an-1])
                    current_S = ""
                    current_A = []

            
            

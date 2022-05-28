from re import L
from turtle import end_fill
import torch
import numpy as np
import torch.nn.functional as F
from model import *
from constants import *
import string 
import matplotlib.pyplot as plt

def get_one_hot(ind, vector_size):
    # return F.one_hot(torch.tensor([ind]), num_classes=vector_size)[0]
    return F.one_hot(torch.tensor([ind]), num_classes=vector_size)[0].type(torch.DoubleTensor)

def get_label_tensor(action, Vocab_size):
    # vector_size = [len(Actions), MAX_SEQ_LEN, MAX_SEQ_LEN, ]
    label = get_one_hot(action[0], len(Actions))
    label = torch.cat(label, get_one_hot(action[1], MAX_TARGET_LEN), get_one_hot(action[2], MAX_TARGET_LEN))

    # after action, start , end index
    for w in action[3:]:
        label = torch.cat(label, get_one_hot(w, Vocab_size))

def correct_punctuations(s):
    for p in string.punctuation:
        s = s.replace(p, " " + p + " ")
    return s

def convert_action2label(action, vocab):
    # ex action string = 7,8,cause
    # ex label = [action_type_index, start_index, end_index, <ind1, ind2 .... indi> from vocab]
    tokens = action.split(",") 
    try :
        start, end = int(tokens[0]), int(tokens[1])
        text = " , ".join([t.strip() for t in tokens[2:]])

        act_ind = 0

        if start ==  end:
            act_ind = 0
        elif len(text)==0:
            act_ind = 2
        else:
            act_ind = 1
        
        word_index = []
        for w in text.split():
            if w not in vocab: 
                print(w, " not in vocab!!")
                w = "<NONE>"
            word_index.append(vocab.index(w))
                
        for i in range(len(text.split()), MAX_TARGET_LEN):
            word_index.append(vocab.index("<NONE>"))
    except:
        word_index = []
        act_ind, start, end = 0, 0, 0
        for i in range(0, MAX_TARGET_LEN):
            word_index.append(vocab.index("<NONE>"))
        # print("Something wrong in convert_action2label \n",action)

    return [act_ind, start, end, word_index]

def get_label_match(action, start_index, end_index, output_seq, label, batch_ind):
    action_acc = int(action[batch_ind, :].argmax() == label[0])
    start_index_acc = int(start_index[batch_ind, :].argmax() == label[1])
    end_index_acc = int(end_index[batch_ind, :].argmax() == label[2])
    output_seq_acc = int(((output_seq[batch_ind, :].argmax(dim=-1).detach().cpu().numpy() == np.asarray(label[3])).all()).all())
    return np.asarray([action_acc, start_index_acc, end_index_acc, output_seq_acc])


def batch_prediction_accuracy(action, start_index, end_index, output_seq, tokenizer, input_ids, sent_actions, vocab):

    partial_acc = np.asarray([0,0,0,0])
        
    for b in range(action.size()[0]):
        sent_id = input_ids[b, :]
        word_sent = tokenizer.decode(sent_id)
        # remove padding and other tokens
        word_sent = correct_punctuations(word_sent[word_sent.index("<s>")+3:word_sent.index("</s>")])
        word_sent = "".join(word_sent.split())
        best_match_arr, best_score = [], -1

        for act in sent_actions[word_sent]:
            label = convert_action2label(act, vocab)
            score_arr = get_label_match(action, start_index, end_index, output_seq, label, b)

            if best_score < score_arr.sum():
                best_score = score_arr.sum()
                best_match_arr = score_arr

        partial_acc = np.vstack((partial_acc, best_match_arr))

    # remove the first extra [0,0,0,0] that was used to intialise 
    partial_acc =  partial_acc[1:, :]

    acc = partial_acc[:, 1:].sum(1) == partial_acc.shape[1]
    return acc.sum(), partial_acc.sum(0)


def action_accuracy(action, start_index, end_index, output_seq, label):

    action_acc = (action.argmax(dim=1) == label[:,0].type(torch.int64))
    action_acc_count = action_acc.sum().item()
    # print(action_acc_count)
    start_index_acc = (start_index.argmax(dim=1) == label[:,1].type(torch.int64))
    start_acc_count = start_index_acc.sum().item()
    # print("Pred start_index_acc - ",start_index.argmax(dim=1))
    # print("GT start_index_acc - ", label[:,1].type(torch.int64))
    # print(start_acc_count)
    
    end_index_acc = (end_index.argmax(dim=1) == label[:,2].type(torch.int64))
    end_acc_count = end_index_acc.sum().item()
    output_seq_acc = (output_seq.argmax(dim=-1) == label[:,3:].type(torch.int64))
    output_seq_numpy = output_seq_acc.detach().cpu().numpy()
    output_acc = (output_seq_numpy.sum(1) == output_seq_numpy.shape[1]).sum()

    acc_matrix = torch.concat((torch.unsqueeze(action_acc, dim=1), torch.unsqueeze(start_index_acc,dim= 1), 
                torch.unsqueeze(end_index_acc, dim = 1), output_seq_acc), dim=1)
    
    all_acc = acc_matrix.detach().cpu().numpy()
    # print("all_acc ", type(all_acc))
    acc = all_acc.sum(1) == all_acc.shape[1]

    # print("Acc = ", acc.sum())
    return acc.sum(), [action_acc_count, start_acc_count, end_acc_count, output_acc]


def get_pred_action(action, start_index, end_index, output_seq, ind, vocab):

    action_label = Actions[action.argmax(dim=1)[ind]]
    start_index = start_index.argmax(dim=1)[ind].item()
    end_index = end_index.argmax(dim=1)[ind].item()

    output_seq = output_seq.detach().cpu().numpy()
    out_str = ""
    # print(output_seq[0, :, :].shape)
    for i in range(output_seq.shape[1]):
        out_str += vocab[np.argmax(output_seq[ind, i, :])] + " "

        # out_str += vocab_10gec[output_seq[i, :].argmax(dim=1)[ind]] + " "

    return (str(action_label) + "\t" + str(start_index) + "\t" + str(end_index) + "\t" + out_str)

def plot_graphs(result_folder, graph_name, train_loss_arr, train_acc_arr, val_loss_arr, val_acc_arr):
    fig, ax = plt.subplots()
    fig.suptitle('Loss and Acc')
    ax.plot(val_loss_arr, label='Val Loss', color='blue')
    ax.plot(train_loss_arr, '--', label='Train Loss', color='lightblue')
    # ax.plot(val_acc_arr, label='Val Acc', color='orange')
    # ax.plot(train_acc_arr, '--', label='Train Acc', color='lightcoral')
    ax.legend(prop={"size": 7}, bbox_to_anchor=(1, 0.5))
    plt.savefig(result_folder + graph_name +".pdf")
    plt.tight_layout()
    plt.close('all')
    # return

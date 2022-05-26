import torch
import numpy as np
import torch.nn.functional as F
from model import *
from constants import *
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

def action_accuracy(action, start_index, end_index, output_seq, label):

    action_acc = (action.argmax(dim=1) == label[:,0].type(torch.int64))
    action_acc_count = action_acc.sum().item()
    start_index_acc = (start_index.argmax(dim=1) == label[:,1].type(torch.int64))
    start_acc_count = start_index_acc.sum().item()
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

    return (str(action_label) + "    " + str(start_index) + "    " + str(end_index) + "    " + out_str)

def plot_graphs(result_folder, graph_name, train_loss_arr, train_acc_arr, val_loss_arr, val_acc_arr):
    fig, ax = plt.subplots()
    fig.suptitle('Loss and Acc')
    ax.plot(val_loss_arr, label='Val Loss', color='blue')
    ax.plot(train_loss_arr, '--', label='Train Loss', color='lightblue')
    ax.plot(val_acc_arr, label='Val Acc', color='orange')
    ax.plot(train_acc_arr, '--', label='Train Acc', color='lightcoral')
    ax.legend(prop={"size": 7}, bbox_to_anchor=(1, 0.5))
    plt.savefig(result_folder + graph_name +".pdf")
    plt.tight_layout()
    plt.close('all')
    # return 

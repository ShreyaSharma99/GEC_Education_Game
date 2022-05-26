from functools import partial
import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizer
from torch import nn
from transformers import BertModel
import torchtext.data 
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from tqdm import tqdm
import torch.nn.functional as F
import json
from model import *
import os
from constants import *
from helper import *

'''
Probable To Dos -
1. Bert vocabulory for prediction vocabulory set
2. Correct the code to handle single batch size data
3. Use BART encoder - decoder model (pre-trained)
4. Use start and end index information as 
    a. Appended as Positional encoder 
    b. As a new token in the embedding
5. Check if the mask 10e32 is right
'''


############################# Load the Dataset and generate the vocabulory #############################
df = pd.read_csv(datapath, nrows=44480)
# 44480)
df.head()

vocab_10gec = set()
for index, row in df.iterrows():
    vocab_10gec.update(set(row['action'].split(",")[-1].strip().split()))
    vocab_10gec.update(set(row['text'].strip().split()))

vocab_10gec = sorted(list(vocab_10gec))
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

# with open('10gec_vocab.json') as json_file:
#     vocab_10gec= json.load(json_file)

vocab_10gec.append("<NONE>")
VOCAB_SIZE = len(vocab_10gec)
print("Vocab_size = ", VOCAB_SIZE)


class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, df, tokenizer):
        
        action_ind = []
        for label in df["action"]:
            ind_list = label.split(",")
            if ind_list[0] == ind_list[1]: #Add
                act = 0
            elif len(ind_list) == 3 and ind_list[2] == "": # delete
                act = 2
            else: # replace
                act = 1
            words = " , ".join(ind_list[2:])
            word_index = []
            for w in words.split():
                if "," in w and len(w)>1:
                    word_index.append(vocab_10gec.index(","))
                    w = w[1:]

                if w not in vocab_10gec: w = "<NONE>"
                word_index.append(vocab_10gec.index(w))
            
            for i in range(len(words.split()), MAX_TARGET_LEN):
                word_index.append(vocab_10gec.index("<NONE>"))
            
            action_ind.append([act] + [int(ind_list[0]), int(ind_list[1])] + word_index)

        # self.labels = [ get_one_hot(act) for act in action_ind]
        self.labels = [torch.Tensor(act) for act in action_ind]
        
        # self.labels = [act for act in df['action']]
        self.texts = [tokenizer(text, padding='max_length', max_length = MAX_INPUT_LEN, truncation=True,
                                return_tensors="pt") for text in df['text']]

        self.sent_text = [text for text in df['text']]
        self.action_text = [label for label in df["action"]]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y

def train(model, tokenizer, train_data, val_data, learning_rate, epochs, result_folder):

    path = os.path.join("model_files")
    train, val = Dataset(train_data, tokenizer), Dataset(val_data, tokenizer)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=BATCH_SIZE)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr= learning_rate)

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    train_acc_arr = []
    val_acc_arr = []
    train_loss_arr = []
    val_loss_arr = []
    
    for epoch_num in range(epochs):

        total_acc_train = 0
        part_acc_train = [0,0,0,0]
        total_loss_train = 0

        for train_input, train_label in tqdm(train_dataloader):
            # print("train_label,s " - train_label)
            train_label = train_label.to(device)
            # print("train_label = ", train_label.size())
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)

            batch_size = train_label.size()[0]
            src_mask = generate_square_subsequent_mask(batch_size).to(device)

            action, start_index, end_index, output_seq = model(input_id, mask, src_mask, batch_size)

            # print("End_pred - ", end_index)
            # print("swapped out axis - ", torch.swapax
            # .size())
                
            # batch_loss = criterion(action, train_label[:, 0].type(torch.int64))/len(Actions)
            batch_loss = criterion(action, train_label[:, 0].type(torch.int64))/len(Actions) + \
                        criterion(start_index, train_label[:, 1].type(torch.int64))/MAX_TARGET_LEN + \
                        criterion(end_index, train_label[:, 2].type(torch.int64))/MAX_TARGET_LEN  + \
                        criterion(torch.swapaxes(output_seq, 1, 2), train_label[:, 3:].type(torch.int64))/VOCAB_SIZE

            total_loss_train += batch_loss.item()
            acc, partial_acc = action_accuracy(action, start_index, end_index, output_seq,train_label)

            # acc = (output.argmax(dim=1) == train_label.argmax(dim=1)).sum().item()
            total_acc_train += acc
            part_acc_train += partial_acc
            # print("Batch_loss = ", batch_loss)
            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

            if epoch_num % 4 == 3:
                torch.save(model.state_dict(), path)
                # model.save_pretrained(path)
                # tokenizer.save_pretrained(path)

        train_loss_arr.append(total_loss_train); train_acc_arr.append(total_acc_train)

        total_acc_val = 0
        part_acc_val = [0,0,0,0]
        total_loss_val = 0

        with torch.no_grad():
            
            predictions_list = []
            for val_input, val_label in val_dataloader:

                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                
                batch_size_val = val_label.size()[0]
                src_mask = generate_square_subsequent_mask(batch_size_val).to(device)

                action, start_index, end_index, output_seq = model(input_id, mask, src_mask, batch_size_val)

                batch_loss = criterion(action, val_label[:, 0].type(torch.int64))/len(Actions) + \
                        criterion(start_index, val_label[:, 1].type(torch.int64))/MAX_TARGET_LEN + \
                        criterion(end_index, val_label[:, 2].type(torch.int64))/MAX_TARGET_LEN + \
                        criterion(torch.swapaxes(output_seq, 1, 2), val_label[:, 3:].type(torch.int64))/VOCAB_SIZE

                total_loss_val += batch_loss.item()

                acc, partial_acc = action_accuracy(action, start_index, end_index, output_seq, val_label)

                total_acc_val += acc
                part_acc_val += partial_acc

                if epoch_num % 4 == 3:
                    for b in range(batch_size_val):
                        pred = get_pred_action(action, start_index, end_index, output_seq, b, vocab_10gec)
                        # pred = val_sent + "\n" + val_action + "\n" + pred + "\n"
                        predictions_list.append(pred)
                        # print(pred)

            if epoch_num % 4 == 3:
                json_string = json.dumps(predictions_list)
                with open('predictions_val.json', 'w') as outfile:
                    json.dump(json_string, outfile)

            val_loss_arr.append(total_loss_val); val_acc_arr.append(total_acc_val)
                
        
        print(
            f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data) : .3f} \
            | Train Accuracy: {total_acc_train / len(train_data): .3f} \
            | Train Partial Accuracy: {part_acc_train[0] / len(train_data): .3f} , {part_acc_train[1] / len(train_data): .3f} , {part_acc_train[2] / len(train_data): .3f} ,  {part_acc_train[3] / len(train_data): .3f} \n \
            | Val Loss: {total_loss_val / len(val_data): .3f} \
            | Val Accuracy: {total_acc_val / len(val_data): .3f} \
            | Val Partial Accuracy: {part_acc_val[0] / len(val_data): .3f} , {part_acc_val[1] / len(val_data): .3f} , {part_acc_val[2] / len(val_data): .3f} ,  {part_acc_val[3] / len(val_data): .3f}')

        if epoch_num%4 == 3:
            print("save graph!!")
            plot_graphs(result_folder, "b1_graph", train_loss_arr, train_acc_arr, val_loss_arr, val_acc_arr)
            
# | Train Partial Accuracy: {np.asarray(part_acc_train) / len(train_data): .3f} \
                

def evaluate(model, tokenizer, test_data):

    test = Dataset(test_data, tokenizer)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=BATCH_SIZE)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:

        model = model.cuda()

    total_acc_test = 0
    part_acc_test = [0,0,0,0]

    predictions_list = []

    with torch.no_grad():

        for test_input, test_label in test_dataloader:

            test_label = test_label.to(device)
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)

            batch_size_test = test_label.size()[0]
            src_mask = generate_square_subsequent_mask(batch_size_test).to(device)

            action, start_index, end_index, output_seq = model(input_id, mask, src_mask, batch_size_test)
            acc, part_acc = action_accuracy(action, start_index, end_index, output_seq,test_label)
        #   acc = (output.argmax(dim=1) == test_label).sum().item()
            total_acc_test += acc
            part_acc_test += part_acc

            for b in range(batch_size_test):
                pred = get_pred_action(action, start_index, end_index, output_seq, b, vocab_10gec)
                # pred = test_sent + "\n" + test_action + "\n" + pred + "\n"
                predictions_list.append(pred)

        json_string = json.dumps(predictions_list)
        with open('predictions_test.json', 'w') as outfile:
            json.dump(json_string, outfile)
    
    print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}\
            | Test Partial Accuracy: {part_acc_test[0] / len(test_data): .3f} , {part_acc_test[1] / len(test_data): .3f} , {part_acc_test[2] / len(test_data): .3f} ,  {part_acc_test[3] / len(test_data): .3f}')
            # Test Part Accuracy: {part_acc_test / len(test_data): .3f}')

ntokens = len(vocab_10gec) # 32139 = len(vocab) + <NONE>  # size of vocabulary
em_size = 768  # embedding dimension
d_hid = 200  # dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2  # number of heads in nn.MultiheadAttention
dropout = 0.2  # dropout probability
embed_size = 768
action_len = len(Actions)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
EPOCHS = 50

print("Size ntokens = ", ntokens)
result_folder = "model_results_b1/"
os.makedirs(result_folder, exist_ok=True)

df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42), 
                                     [int(.8*len(df)), int(.9*len(df))])

# print("df.size() = ", len(df))

model = BertClassifier(MAX_INPUT_LEN, MAX_TARGET_LEN, embed_size, action_len, ntokens, em_size, nhead, d_hid, nlayers, dropout).to(device)

LR = 1e-6

# print("df_train - ", df_train.loc[0])
train(model, tokenizer, df_train, df_val, LR, EPOCHS, result_folder)

evaluate(model, tokenizer, df_test)
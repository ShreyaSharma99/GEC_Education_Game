from tokenize import single_quoted
import torch
import math
from torch import nn, Tensor
from transformers import BertModel
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from transformers import BartModel

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


# # 2. Add layers to get action_index, Start and end index, seq2seq transformer for word generation
# class BertClassifier(nn.Module):

#     def __init__(self, max_input_len, max_target_len, embed_size, action_len, ntoken: int, d_model: int, nhead: int, d_hid: int,
#                  nlayers: int, dropout: float = 0.5):

#         super(BertClassifier, self).__init__()

#         self.bert = BertModel.from_pretrained('bert-base-cased')
#         self.dropout = nn.Dropout(dropout)

#         # action prediction = maps [batch_size X embed_size]  -> [batch_size X 6]
#         self.action_linear = nn.Linear(embed_size, action_len)

#         self.start_linear1 = nn.Linear(embed_size, 1)
#         self.start_linear2 = nn.Linear(max_input_len, max_target_len)

#         self.end_linear1 = nn.Linear(embed_size, 1)
#         self.end_linear2 = nn.Linear(max_input_len, max_target_len)

#         # start prediction = maps [batch_size X seq_len X embed_size]  -> [batch_size X seq_len X 1]
#         self.relu = nn.ReLU()

#         self.softmax = nn.Softmax(dim=1)

#         # transformer layers
#         self.pos_encoder = PositionalEncoding(d_model, dropout)
#         encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
#         self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
#         # self.encoder = nn.Embedding(ntoken, d_model)
#         self.d_model = d_model
#         self.decoder = nn.Linear(d_model, ntoken)

#         self.init_weights()

#     def init_weights(self) -> None:
#         initrange = 0.1
#         # self.encoder.weight.data.uniform_(-initrange, initrange)
#         self.decoder.bias.data.zero_()
#         self.decoder.weight.data.uniform_(-initrange, initrange)
        

#     def forward(self, input_id, mask, src_mask, batch_size):

#         seq_output, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)

#         dropout_output = self.dropout(pooled_output)
#         act_linear_output = self.action_linear(dropout_output)
#         action = self.softmax(act_linear_output)

#         dropout_output = self.dropout(seq_output)
        
#         start_index = self.softmax(torch.squeeze(self.start_linear1(dropout_output), 2))
#         # start_index_l2 = self.start_linear2(torch.squeeze(self.start_linear1(dropout_output)))
#         # start_index = self.softmax(start_index_l2)

#         start_mask = torch.zeros_like(start_index)
#         index = int(torch.argmax(start_index, dim=1)[0])
#         start_mask[:, 0:index] = float("-inf")

#         end_index = self.softmax(torch.squeeze(self.end_linear1(dropout_output), 2) + start_mask)

#         # src = self.encoder(seq_output) * math.sqrt(self.d_model)
#         src = self.pos_encoder(seq_output)
#         output = self.transformer_encoder(src, src_mask)
#         output_seq = self.decoder(output)


#         return action, start_index, end_index, output_seq





# # BART Model with Transformer Decoder
# class Bart_Transformer(nn.Module):

#     def __init__(self, max_input_len, max_target_len, embed_size, action_len, ntoken: int, d_model: int, nhead: int, d_hid: int,
#                  nlayers: int, dropout: float = 0.5):

#         super(Bart_Transformer, self).__init__()

#         self.bart = BartModel.from_pretrained("facebook/bart-base")
#         self.dropout = nn.Dropout(dropout)

#         # action prediction = maps [batch_size X embed_size]  -> [batch_size X 6]
#         self.action_linear = nn.Linear(embed_size, action_len)

#         self.start_linear1 = nn.Linear(embed_size, 1)
#         self.start_linear2 = nn.Linear(max_input_len, max_target_len)

#         self.end_linear1 = nn.Linear(embed_size, 1)
#         self.end_linear2 = nn.Linear(max_input_len, max_target_len)

#         self.softmax = nn.Softmax(dim=1)

#         self.pos_encoder = PositionalEncoding(d_model, dropout)
#         encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
#         self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
#         # self.encoder = nn.Embedding(ntoken, d_model)
#         self.d_model = d_model
#         self.decoder = nn.Linear(d_model, ntoken)

#         self.init_weights()

#     def init_weights(self) -> None:
#         initrange = 0.1
#         # self.encoder.weight.data.uniform_(-initrange, initrange)
#         self.decoder.bias.data.zero_()
#         self.decoder.weight.data.uniform_(-initrange, initrange)
        
        

#     def forward(self, input_id, mask, src_mask, batch_size):

#         output = self.bart(input_ids= input_id, attention_mask = mask)

#         seq_output = output.encoder_last_hidden_state
#         pooled_output = torch.mean(output.last_hidden_state, 1)

#         dropout_output = self.dropout(pooled_output)
#         act_linear_output = self.action_linear(dropout_output)
#         action = self.softmax(act_linear_output)

#         dropout_output = self.dropout(seq_output)
        
#         start_index = self.softmax(torch.squeeze(self.start_linear1(dropout_output), 2))
        
#         start_mask = torch.zeros_like(start_index)
#         index = int(torch.argmax(start_index, dim=1)[0])
#         start_mask[:, 0:index] = float("-inf")

#         end_index = self.softmax(torch.squeeze(self.end_linear1(dropout_output), 2) + start_mask)

#         src = self.pos_encoder(output.last_hidden_state)
#         output_transformer = self.transformer_encoder(src, src_mask)
#         output_seq = self.decoder(output_transformer)

#         # output_seq = self.decoder(output.last_hidden_state)
        
#         return action, start_index, end_index, output_seq


# BART Model with Transformer Decoder and start and end index encoding
class Bart_Transformer_SE(nn.Module):

    def __init__(self, max_input_len, max_target_len, embed_size, action_len, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):

        super(Bart_Transformer_SE, self).__init__()

        self.bart = BartModel.from_pretrained("facebook/bart-base")
        self.dropout = nn.Dropout(dropout)

        # action prediction = maps [batch_size X embed_size]  -> [batch_size X 6]
        self.action_linear = nn.Linear(embed_size, action_len)

        self.start_linear1 = nn.Linear(embed_size, 1)
        self.start_linear2 = nn.Linear(max_input_len, max_target_len)

        self.end_linear1 = nn.Linear(embed_size, 1)
        self.end_linear2 = nn.Linear(max_input_len, max_target_len)

        self.softmax = nn.Softmax(dim=1)

        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        # self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        # self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
        

    def forward(self, input_id, mask, src_mask, batch_size):

        output = self.bart(input_ids= input_id, attention_mask = mask)

        seq_output = output.encoder_last_hidden_state
        pooled_output = torch.mean(output.last_hidden_state, 1)

        dropout_output = self.dropout(pooled_output)
        act_linear_output = self.action_linear(dropout_output)
        action = self.softmax(act_linear_output)

        dropout_output = self.dropout(seq_output)
        
        start_index = self.softmax(torch.squeeze(self.start_linear1(dropout_output), 2))
        # print("start_index ", start_index.size())
        start_mask = torch.zeros_like(start_index)
        s_index = torch.argmax(start_index, dim=1)
        # print(s_index.size())
        for i in range(batch_size):
            start_mask[i][0:s_index[i]] = float("-inf")

        end_index = self.softmax(torch.squeeze(self.end_linear1(dropout_output), 2) + start_mask)
        e_index = torch.argmax(end_index, dim=1)

        src = self.pos_encoder(output.last_hidden_state)

        start_embed = torch.zeros_like(src[0, 0:1, :])
        

        for i in range(src.size()[0]):
            start_index, end_index = s_index[i], e_index[i]
            if start_index >= src.size()[1]-2: start_index = src.size()[1]-3
            if end_index >= src.size()[1]-2: end_index = src.size()[1]-3 
            src[i] = torch.cat((src[i][:start_index], start_embed, src[i][start_index : end_index], start_embed, src[i][e_index[i]:-2]), 0)

        print("ssrc size = ", src.size())

        output_transformer = self.transformer_encoder(src, src_mask)
        output_seq = self.decoder(output_transformer)

        # output_seq = self.decoder(output.last_hidden_state)
        
        return action, start_index, end_index, output_seq




# # BART
# class BartSeq2Seq(nn.Module):

#     def __init__(self, max_input_len, max_target_len, embed_size, action_len, ntoken: int, d_model: int, nhead: int, d_hid: int,
#                  nlayers: int, dropout: float = 0.5):

#         super(BartSeq2Seq, self).__init__()

#         self.bart = BartModel.from_pretrained("facebook/bart-base")
#         self.dropout = nn.Dropout(dropout)

#         # action prediction = maps [batch_size X embed_size]  -> [batch_size X 6]
#         self.action_linear = nn.Linear(embed_size, action_len)

#         self.start_linear1 = nn.Linear(embed_size, 1)
#         self.start_linear2 = nn.Linear(max_input_len, max_target_len)

#         self.end_linear1 = nn.Linear(embed_size, 1)
#         self.end_linear2 = nn.Linear(max_input_len, max_target_len)

#         self.softmax = nn.Softmax(dim=1)

#         self.d_model = d_model
#         self.decoder = nn.Linear(d_model, ntoken)

#         self.init_weights()

#     def init_weights(self) -> None:
#         initrange = 0.1
#         # self.encoder.weight.data.uniform_(-initrange, initrange)
#         self.decoder.bias.data.zero_()
#         self.decoder.weight.data.uniform_(-initrange, initrange)
        

#     def forward(self, input_id, mask, src_mask, batch_size):

#         output = self.bart(input_ids= input_id, attention_mask = mask)

#         seq_output = output.encoder_last_hidden_state
#         pooled_output = torch.mean(output.last_hidden_state, 1)

#         dropout_output = self.dropout(pooled_output)
#         act_linear_output = self.action_linear(dropout_output)
#         action = self.softmax(act_linear_output)

#         dropout_output = self.dropout(seq_output)
        
#         start_index = self.softmax(torch.squeeze(self.start_linear1(dropout_output), 2))
        
#         start_mask = torch.zeros_like(start_index)
#         index = int(torch.argmax(start_index, dim=1)[0])
#         start_mask[:, 0:index] = float("-inf")

#         end_index = self.softmax(torch.squeeze(self.end_linear1(dropout_output), 2) + start_mask)

#         output_seq = self.decoder(output.last_hidden_state)
        
#         return action, start_index, end_index, output_seq

import torch
import math
from torch import nn, Tensor
from transformers import BertModel
from torch.nn import TransformerEncoder, TransformerEncoderLayer

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


# 2. Add layers to get action_index, Start and end index, seq2seq transformer for word generation
class BertClassifier(nn.Module):

    def __init__(self, max_input_len, max_target_len, embed_size, action_len, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)

        # action prediction = maps [batch_size X embed_size]  -> [batch_size X 6]
        self.action_linear = nn.Linear(embed_size, action_len)

        self.start_linear1 = nn.Linear(embed_size, 1)
        self.start_linear2 = nn.Linear(max_input_len, max_target_len)

        self.end_linear1 = nn.Linear(embed_size, 1)
        self.end_linear2 = nn.Linear(max_input_len, max_target_len)

        # start prediction = maps [batch_size X seq_len X embed_size]  -> [batch_size X seq_len X 1]
        # self.index_linear = nn.Linear(embed_size, max_seq_len)

        self.relu = nn.ReLU()

        self.softmax = nn.Softmax(dim=1)

        # transformer layers
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

        seq_output, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)

        dropout_output = self.dropout(pooled_output)
        act_linear_output = self.action_linear(dropout_output)
        action = self.softmax(act_linear_output)

        dropout_output = self.dropout(seq_output)
        
        start_index = self.softmax(torch.squeeze(self.start_linear1(dropout_output), 2))
        # start_index_l2 = self.start_linear2(torch.squeeze(self.start_linear1(dropout_output)))

        # start_index = self.softmax(start_index_l2)
        start_mask = torch.zeros_like(start_index)

        # print("action - ", action.size())
        # print("start - ", start_index.size())

        index = int(torch.argmax(start_index, dim=1)[0])
        start_mask[:, 0:index] = float("-inf")

        end_index = self.softmax(torch.squeeze(self.end_linear1(dropout_output), 2) + start_mask)
        # end_index = self.softmax(self.start_linear2(torch.squeeze(self.end_linear1(dropout_output))) + start_mask)
                    # OR #
        # do the projection and multiply
        
        # src = self.encoder(seq_output) * math.sqrt(self.d_model)
        src = self.pos_encoder(seq_output)
        output = self.transformer_encoder(src, src_mask)

        # print("output size = ", output.size())

        output_seq = self.decoder(output)

        # print("Decoder seq - ", output_seq.size())

        return action, start_index, end_index, output_seq

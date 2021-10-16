import torch.nn as nn
from supar.modules import LSTM
from supar.modules.dropout import IndependentDropout, SharedDropout
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMencoder(nn.Module):
    def __init__(self, conf, input_dim, **kwargs):
        super(LSTMencoder, self).__init__()
        self.conf = conf

        self.before_lstm_dropout = None

        if self.conf.embed_dropout_type == 'independent':
            self.embed_dropout = IndependentDropout(p=conf.embed_dropout)
            if conf.before_lstm_dropout:
                self.before_lstm_dropout = SharedDropout(p=conf.before_lstm_dropout)

        elif self.conf.embed_dropout_type == 'shared':
            self.embed_dropout = SharedDropout(p=conf.embed_dropout)

        elif self.conf.embed_dropout_type == 'simple':
            self.embed_dropout = nn.Dropout(p=conf.embed_dropout)

        else:
            self.embed_dropout = nn.Dropout(0.)

        self.lstm = LSTM(input_size=input_dim,
                         hidden_size=conf.n_lstm_hidden,
                         num_layers=conf.n_lstm_layers,
                         bidirectional=True,
                         dropout=conf.lstm_dropout)
        self.lstm_dropout = SharedDropout(p=conf.lstm_dropout)



    def forward(self, info):
        # lstm encoder
        embed = info['embed']
        seq_len = info['seq_len']

        embed = [i for i in embed.values()]

        if self.conf.embed_dropout_type == 'independent':
            embed = self.embed_dropout(embed)
            embed = torch.cat(embed, dim=-1)
        else:
            embed = torch.cat(embed, dim=-1)
            embed = self.embed_dropout(embed)

        seq_len = seq_len.cpu()
        x = pack_padded_sequence(embed, seq_len.cpu() + (embed.shape[1] - seq_len.max()), True, False)
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, True, total_length=embed.shape[1])
        x = self.lstm_dropout(x)

        info['encoded_emb'] = x

    def get_output_dim(self):
        return self.conf.n_lstm_hidden * 2


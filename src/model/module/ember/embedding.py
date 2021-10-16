import torch.nn as nn
import torch
from supar.modules.char_lstm import CharLSTM
from supar.modules import TransformerEmbedding
import copy

class Embeder(nn.Module):
    def __init__(self, conf, fields):
        super(Embeder, self).__init__()
        self.conf = conf

        if 'pos' in fields.inputs:
            self.pos_emb = nn.Embedding(fields.get_vocab_size("pos"), conf.n_pos_embed)
        else:
            self.pos_emb = None

        if 'char' in fields.inputs:
            self.feat = CharLSTM(n_chars=fields.get_vocab_size('char'),
                                       n_embed=conf.n_char_embed,
                                       n_out=conf.n_char_out,
                                      pad_index=fields.get_pad_index('char'),
                                        input_dropout=conf.char_input_dropout)
            self.feat_name = 'char'

        if 'bert' in fields.inputs:
            self.feat =  TransformerEmbedding(model=fields.get_bert_name(),
                                            n_layers=conf.n_bert_layers,
                                            n_out=conf.n_bert_out,
                                            pad_index=fields.get_pad_index("bert"),
                                            dropout=conf.mix_dropout,
                                            requires_grad=conf.finetune,
                                            use_projection=conf.use_projection,
                                            use_scalarmix=conf.use_scalarmix)
            self.feat_name = "bert"
            print(fields.get_bert_name())

        if  ('char' not in fields.inputs and 'bert' not in fields.inputs):
            self.feat = None

        if 'word' in fields.inputs:
            ext_emb = fields.get_ext_emb()
            if ext_emb:
                self.word_emb =  copy.deepcopy(ext_emb)
            else:
                self.word_emb = nn.Embedding(num_embeddings=fields.get_vocab_size('word'),
                                             embedding_dim=conf.n_embed)
        else:
            self.word_emb = None


    def forward(self, ctx):
        emb = {}

        if self.pos_emb:
            emb['pos'] = self.pos_emb(ctx['pos'])

        if self.word_emb:
            emb['word'] = self.word_emb(ctx['word'])

        #For now, char or ber、t, choose one.
        if self.feat:
            emb[self.feat_name] = self.feat(ctx[self.feat_name])

        ctx['embed'] = emb


    def get_output_dim(self):

        size = 0

        if self.pos_emb:
            size += self.conf.n_pos_embed

        if self.word_emb:
            if isinstance(self.word_emb, nn.Embedding):
                size += self.conf.n_embed
            else:
                size += self.word_emb.get_dim()

        if self.feat:
            if self.feat_name == 'char':
                size += self.conf.n_char_out
            else:
                size += self.feat.n_out
        return size




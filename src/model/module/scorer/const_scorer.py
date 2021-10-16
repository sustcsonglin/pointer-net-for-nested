import torch.nn as nn
from .module.biaffine import BiaffineScorer
import torch



class ConstScorer(nn.Module):
    def __init__(self, conf, fields, input_dim):
        super(ConstScorer, self).__init__()
        self.conf = conf

        if self.conf.use_span:
            self.span_scorer = BiaffineScorer(n_in=input_dim, n_out=conf.n_mlp_span, bias_x=True, bias_y=False, dropout=conf.mlp_dropout)
        self.label_scorer = BiaffineScorer(n_in=input_dim, n_out=conf.n_mlp_label, bias_x=True, bias_y=True, dropout=conf.mlp_dropout, n_out_label=fields.get_vocab_size("chart"))
        self.null_idx = fields.get_vocab('chart')['NULL']

        # if self.conf.use_transition:
        # using transition scores?
        # vocab_size = fields.get_vocab_size('chart')
        # self.transition = nn.Parameter(torch.rand(vocab_size, vocab_size))


    def forward(self, ctx):
        x = ctx['encoded_emb']

        if 'span_repr' not in ctx:
            x_f, x_b = x.chunk(2, -1)
            x = torch.cat((x_f[:, :-1], x_b[:, 1:]), -1)
        else:
            x = ctx['span_repr']

        s_span = self.label_scorer(x)
        if self.conf.use_span:
            s_span += self.span_scorer(x).unsqueeze(-1)
        mask = s_span.new_zeros(s_span.shape[1], s_span.shape[1], dtype=torch.bool)
        mask.diagonal(1).fill_(1)
        s_span[..., self.null_idx].masked_fill_(~mask.unsqueeze(0).expand(s_span.shape[0], s_span.shape[1], s_span.shape[1]), -1e9)
        ctx['s_span'] = s_span



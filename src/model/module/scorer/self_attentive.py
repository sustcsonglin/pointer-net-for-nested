import torch.nn as nn
import torch
from .module.biaffine import BiaffineScorer


class LabelMLPScorer(nn.Module):
    def __init__(self, conf, fields, input_dim):
        super(LabelMLPScorer, self).__init__()
        self.conf = conf
        self.f_label = nn.Sequential(
        nn.Linear(input_dim, conf.d_label_hidden),
        nn.LayerNorm(conf.d_label_hidden),
        nn.ReLU(),
        nn.Linear(conf.d_label_hidden, fields.get_vocab_size('chart')),
        )

    def forward(self, ctx):
        fence_post = ctx['fencepost']
        span_repr = fence_post.unsqueeze(1) - fence_post.unsqueeze(2)
        ctx['s_span'] =  self.f_label(span_repr)



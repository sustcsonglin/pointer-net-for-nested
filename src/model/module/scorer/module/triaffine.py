import torch.nn as nn
from supar.modules import  MLP, Triaffine

class TriaffineScorer(nn.Module):
    def __init__(self, n_in=800, n_out=400, bias_x=True, bias_y=False, dropout=0.33):
        super(TriaffineScorer, self).__init__()
        self.l = MLP(n_in=n_in, n_out=n_out, dropout=dropout)
        self.m = MLP(n_in=n_in, n_out=n_out, dropout=dropout)
        self.r = MLP(n_in=n_in, n_out=n_out, dropout=dropout)
        self.attn = Triaffine(n_in=n_out, bias_x=bias_x, bias_y=bias_y)

    def forward(self, h):
        left = self.l(h)
        mid =  self.m(h)
        right = self.r(h)
        #sib, dependent, head)
        return self.attn(left, mid, right).permute(0, 2, 3, 1)

    def forward2(self, word, span):
        left = self.l(word)
        mid = self.m(span)
        right = self.r(span)
        #  head, left_bdr, right_bdr: used in span-head model?

        # word, left, right
        # fine.
        return self.attn(mid, right, left).permute(0, 2, 3, 1)



# class TriaffineScorer
# class TriaffineScorer(nn.Module):






import torch.nn as nn
from supar.modules import  MLP, Biaffine

class BiaffineScorer(nn.Module):
    def __init__(self, n_in=800, n_out=400, n_out_label=1, bias_x=True, bias_y=False, scaling=False,  dropout=0.33):
        super(BiaffineScorer, self).__init__()
        self.l = MLP(n_in=n_in, n_out=n_out, dropout=dropout)
        self.r = MLP(n_in=n_in, n_out=n_out, dropout=dropout)
        self.attn = Biaffine(n_in=n_out, n_out=n_out_label,  bias_x=bias_x, bias_y=bias_y)
        self.scaling = 0 if not scaling else  n_out ** (1/4)
        self.n_in = n_in

    def forward(self, h):
        left = self.l(h)
        right = self.r(h)

        if self.scaling:
            left = left / self.scaling
            right = right / self.scaling

        return self.attn(left, right)


    def forward_v2(self, h, q):
        left = self.l(h)
        right = self.r(q)
        if self.scaling:
            left = left / self.scaling
            right = right / self.scaling

        return self.attn(left, right)

    # def forward_v3(self, h, q):

    def forward_v3(self, h, q):
        src = self.l(h)
        dec = self.r(q)
        return self.attn.forward_v2(src, dec)

    def forward_linear(self, h, q):
        src = self.l(h)
        dec = self.r(q)
        return self.attn.forward2(src, dec)


class BiaffineScorer2(nn.Module):
    def __init__(self, n_in_a=800, n_in_b=800, n_out=400, n_out_label=1, bias_x=False, bias_y=False, scaling=False,  dropout=0.33):
        super(BiaffineScorer2, self).__init__()
        self.l = MLP(n_in=n_in_a, n_out=n_out, dropout=dropout)
        self.r = MLP(n_in=n_in_b, n_out=n_out, dropout=dropout)
        self.attn = Biaffine(n_in=n_out, n_out=n_out_label,  bias_x=bias_x, bias_y=bias_y)
        self.scaling = 0 if not scaling else  n_out ** (1/4)

    def forward(self, h, q):
        src = self.l(h)
        dec = self.r(q)
        return self.attn.forward_v2(src, dec)

    def forward2(self, h, q):
        src = self.l(h)
        dec = self.r(q)
        return self.attn.forward_v3(src, dec)


from supar.utils.alg import kmeans
from supar.utils.data import Sampler
from fastNLP.core.field import Padder
import numpy as np




def get_sampler(lengths, max_tokens, n_buckets, shuffle=True, distributed=False, evaluate=False):
    buckets = dict(zip(*kmeans(lengths, n_buckets)))
    return Sampler(buckets=buckets,
                   batch_size=max_tokens,
                   shuffle=shuffle,
                   distributed=distributed,
                   evaluate=evaluate)




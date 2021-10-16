# -*- coding: utf-8 -*-

from collections import Counter

import torch
from supar.utils.fn import pad
from supar.utils.vocab import Vocab
# from benepar.decode_chart import get_labeled_spans

# 魔改自supar.

class RawField(object):
    r"""
    Defines a general datatype.

    A :class:`RawField` object does not assume any property of the datatype and
    it holds parameters relating to how a datatype should be processed.

    Args:
        name (str):
            The name of the field.
        fn (function):
            The function used for preprocessing the examples. Default: ``None``.
    """

    def __init__(self, name, fn=None):
        self.name = name
        self.fn = fn

    def __repr__(self):
        return f"({self.name}): {self.__class__.__name__}()"

    def preprocess(self, sequence):
        return self.fn(sequence) if self.fn is not None else sequence

    def transform(self, sequence):
        return self.preprocess(sequence)


class Field(RawField):
    r"""
    Defines a datatype together with instructions for converting to :class:`~torch.Tensor`.
    :class:`Field` models common text processing datatypes that can be represented by tensors.
    It holds a :class:`Vocab` object that defines the set of possible values
    for elements of the field and their corresponding numerical representations.
    The :class:`Field` object also holds other parameters relating to how a datatype
    should be numericalized, such as a tokenization method.

    Args:
        name (str):
            The name of the field.
        pad_token (str):
            The string token used as padding. Default: ``None``.
        unk_token (str):
            The string token used to represent OOV words. Default: ``None``.
        bos_token (str):
            A token that will be prepended to every example using this field, or ``None`` for no `bos_token`.
            Default: ``None``.
        eos_token (str):
            A token that will be appended to every example using this field, or ``None`` for no `eos_token`.
        lower (bool):
            Whether to lowercase the text in this field. Default: ``False``.
        use_vocab (bool):
            Whether to use a :class:`Vocab` object. If ``False``, the data in this field should already be numerical.
            Default: ``True``.
        tokenize (function):
            The function used to tokenize strings using this field into sequential examples. Default: ``None``.
        fn (function):
            The function used for preprocessing the examples. Default: ``None``.
    """

    def __init__(self, name, pad=None, unk=None, bos=None, eos=None,
                 lower=False, use_vocab=True, tokenize=None, fn=None, min_freq=1):
        self.name = name
        self.pad = pad
        self.unk = unk
        self.bos = bos
        self.eos = eos
        self.lower = lower
        self.use_vocab = use_vocab
        self.tokenize = tokenize
        self.fn = fn
        self.min_freq = min_freq
        self.specials = [token for token in [pad, unk, bos, eos] if token is not None]

    def __repr__(self):
        s, params = f"({self.name}): {self.__class__.__name__}(", []
        if self.pad is not None:
            params.append(f"pad={self.pad}")
        if self.unk is not None:
            params.append(f"unk={self.unk}")
        if self.bos is not None:
            params.append(f"bos={self.bos}")
        if self.eos is not None:
            params.append(f"eos={self.eos}")
        if self.lower:
            params.append(f"lower={self.lower}")
        if not self.use_vocab:
            params.append(f"use_vocab={self.use_vocab}")
        s += ", ".join(params)
        s += ")"

        return s

    def __getstate__(self):
        state = dict(self.__dict__)
        if self.tokenize is None:
            state['tokenize_args'] = None
        elif self.tokenize.__module__.startswith('transformers'):
            state['tokenize_args'] = (self.tokenize.__module__, self.tokenize.__self__.name_or_path)
            state['tokenize'] = None
        return state

    def __setstate__(self, state):
        tokenize_args = state.pop('tokenize_args', None)
        if tokenize_args is not None and tokenize_args[0].startswith('transformers'):
            from transformers import AutoTokenizer
            state['tokenize'] = AutoTokenizer.from_pretrained(tokenize_args[1]).tokenize
        self.__dict__.update(state)

    @property
    def pad_index(self):
        if self.pad is None:
            return 0
        if hasattr(self, 'vocab'):
            return self.vocab[self.pad]
        return self.specials.index(self.pad)

    @property
    def unk_index(self):
        if self.unk is None:
            return 0
        if hasattr(self, 'vocab'):
            return self.vocab[self.unk]
        return self.specials.index(self.unk)

    @property
    def bos_index(self):
        if hasattr(self, 'vocab'):
            return self.vocab[self.bos]
        return self.specials.index(self.bos)

    @property
    def eos_index(self):
        if hasattr(self, 'vocab'):
            return self.vocab[self.eos]
        return self.specials.index(self.eos)

    @property
    def device(self):
        return 'cuda' if torch.cuda.is_available() else 'cpu'

    def preprocess(self, sequence):
        r"""
        Loads a single example using this field, tokenizing if necessary.
        The sequence will be first passed to ``dm_util`` if available.
        If ``tokenize`` is not None, the input will be tokenized.
        Then the input will be lowercased optionally.

        Args:
            sequence (list):
                The sequence to be preprocessed.

        Returns:
            A list of preprocessed sequence.
        """

        if self.fn is not None:
            sequence = self.fn(sequence)
        if self.tokenize is not None:
            sequence = self.tokenize(sequence)
        if self.lower:
            sequence = [str.lower(token) for token in sequence]

        return sequence

    def build(self, sequences):

        r"""
        Constructs a :class:`Vocab` object for this field from the dataset.
        If the vocabulary has already existed, this function will have no effect.

        Args:
            dataset (Dataset):
                A :class:`Dataset` object. One of the attributes should be named after the name of this field.
        """
        if hasattr(self, 'vocab') or not self.use_vocab:
            return
        # sequences = getattr(dataset, self.name)
        counter = Counter(token
                          for seq in sequences
                          for token in self.preprocess(seq))
        self.vocab = Vocab(counter, self.min_freq, self.specials, self.unk_index)


    def transform(self, sequence):
        sequence = self.preprocess(sequence)
        if self.use_vocab:
            sequence = self.vocab[sequence]
        if self.bos:
            sequence = [self.bos_index] + sequence
        if self.eos:
            sequence = sequence + [self.eos_index]
        return sequence


class SubwordField(Field):

    def __init__(self, *args, **kwargs):
        self.fix_len = kwargs.pop('fix_len') if 'fix_len' in kwargs else 0
        self.subword_bos = kwargs.pop('subword_bos') if 'subword_bos' in kwargs else None
        self.subword_eos = kwargs.pop('subword_eos') if 'subword_eos' in kwargs else None
        if self.fix_len == -1:
            self.fix_len = 100000000
        super().__init__(*args, **kwargs)

        # for charlstm
        if self.subword_bos:
            self.specials.append(self.subword_bos)
        if self.subword_eos:
            self.specials.append(self.subword_eos)


    def build(self, sequences):
        if hasattr(self, 'vocab') or not self.use_vocab:
            return
        counter = Counter(piece
                          for seq in sequences
                          for token in seq
                          for piece in self.preprocess(token))
        self.vocab = Vocab(counter, self.min_freq, self.specials, self.unk_index)


    def transform(self, seq):
        seq = [self.preprocess(token) for token in seq]

        if self.use_vocab:
            seq =  [  [self.vocab[i] if i in self.vocab else self.unk_index for i in token] if token else [self.unk_index]
                 for token in seq]

        if self.bos:
            seq = [[self.bos_index] ] + seq

        if self.eos:
            seq = seq + [[self.eos_index]]

        if self.subword_bos:
            seq =  [ [self.vocab[self.subword_bos]] +  s  for s in seq]

        if self.subword_eos:
            seq = [s + [self.vocab[self.subword_eos]] for s in seq]

        l = min(self.fix_len, max(len(ids) for ids in seq))
        seq = [ids[: l] for ids in seq]
        return seq


def identity(x):
    return x

class SpanField(Field):
    def __init__(self, *args, **kwargs):
        fn = kwargs.pop('fn') if 'fn' in kwargs else identity
        # if fn is None:
            # fn = get_labeled_spans

        self.no_label = kwargs.pop('no_label') if 'no_label' in kwargs else None

        super().__init__(*args, **kwargs, fn=fn)
        if self.no_label:
            self.specials.append(self.no_label)


    def preprocess(self, sequence, building_vocab=False):
        if building_vocab:
            labels = []
            for _, _, label in self.fn(sequence):
                labels.append(label)
            return labels
        else:
            spans = []
            for start, end, label in self.fn(sequence):
                spans.append([start, end, label])
            return spans

    def build(self, sequences, min_freq=1):
        counter = Counter(row
                          for chart in sequences
                          for row in self.preprocess(chart, building_vocab=True)
                        )
        self.vocab = Vocab(counter, min_freq, self.specials, self.unk_index)

    def transform(self, tree):
        spans = self.preprocess(tree, building_vocab=False)
        if self.use_vocab:
            spans = [[span[0], span[1], self.vocab[span[2]]] for span in spans]
        return spans


from supar.utils.common import *
from .dm_util.fields import SubwordField, Field, SpanField
import nltk
from fastNLP.core.dataset import DataSet
import logging
from .base import DataModuleBase
from .dm_util.padder import SpanPadder, SpanLabelPadder
from .trees import load_trees,  tree2span, get_nongold_span
from src.datamodule.benepar.decode_chart import get_labeled_spans
from functools import cmp_to_key
from .trees import transition_system

log = logging.getLogger(__name__)

class ConstData4Pointer(DataModuleBase):
    def __init__(self, conf):
        super(ConstData4Pointer, self).__init__(conf)

    def get_inputs(self):
        return ['seq_len', 'chart',  'action_length', 'action_golds', 'span_start', 'span_end']

    def get_targets(self):
        return ['raw_tree', ]

    def build_datasets(self):
        datasets = {}
        conf = self.conf
        datasets['train'] = self._load(const_file=conf.train_const)
        # if not self.conf.debug:
        datasets['dev'] = self._load(const_file=conf.dev_const)
        datasets['test'] = self._load(const_file=conf.test_const)
        return datasets

    def _load(self,  const_file):
        log.info(f'loading: {const_file}')
        dataset = DataSet()
        with open(const_file, encoding='utf-8') as f:
            raw_treebank = [line.rstrip() for line in f]

        trees, word, pos, raw_tree = get_pos_word_from_raw_tree(raw_treebank)
        spans = [get_labeled_spans(tree) for tree in trees]
        length = [len(w) for w in word]

        span_start = []
        span_end = []
        label = []
        action_golds = []


        for idx, (s, l) in enumerate(zip(spans, length)):
            s = process(s, l, use_fine_grained=True)
            ss, se, sl, g =  transition_system(s, l)
            span_start.append(ss)
            span_end.append(se)
            label.append(sl)
            action_golds.append(g)

        dataset.add_field('word', word)
        dataset.add_field('pos', pos)

        action_length = [len(a) for a in action_golds]

        dataset.add_field('action_length', action_length, )
        dataset.add_field('action_golds', action_golds,)
        dataset.add_field('chart', label)
        dataset.add_field('span_start', span_start)
        dataset.add_field('span_end', span_end)
        dataset.add_field('raw_tree', raw_tree, ignore_type=True, padder=None)

        #place holder
        dataset.add_field('char', word)
        dataset.add_field('raw_word', word)
        dataset.add_field('raw_raw_word', word)
        dataset.add_seq_len("raw_word", 'seq_len')
        log.info(f'loading: {const_file} finished')
        return dataset

    #
    def _set_padder(self, datasets):
        pass

    def build_fields(self, train_data):
        fields = {}
        fields['word'] = Field('word', pad=PAD, unk=UNK, bos=BOS, eos=EOS, lower=True, min_freq=self.conf.min_freq)
        fields['pos'] = Field('pos', pad=PAD, unk=UNK, bos=BOS, eos=EOS)
        fields['char'] = SubwordField('char', pad=PAD, unk=UNK, bos=BOS, eos=EOS, subword_eos=subword_eos, subword_bos=subword_bos,
                                      fix_len=self.conf.fix_len)
        fields['chart'] = Field('chart', pad=PAD, unk=UNK)

        for name, field in fields.items():
            field.build(train_data[name])
        return fields

def get_pos_word_from_raw_tree(raw_treebank):
    trees = []
    word = []
    pos = []
    tree_string = []
    for s in raw_treebank:
        if '(TOP' not in s:
            s = '(TOP ' + s + ')'
        tree = nltk.Tree.fromstring(s)
        w, p = zip(*tree.pos())
        word.append(w)
        pos.append(p)
        trees.append(tree)
        tree_string.append(s)
    return trees, word, pos, tree_string



def process(spans, length, use_fine_grained=False):
    def compare(a, b):
        if a[0] > b[0]:
            return 1
        elif a[0] == b[0]:
            if a[1] > b[1]:
                return -1
            else:
                return 1
        else:
            return -1

    def compare2(a, b):
        if int(a[0]) >= int(b[1]):
            return 1

        elif b[0] <= a[0] <= a[1] <= b[1]:
            return -1

        elif a[0] <= b[0] <= b[1] <= a[1]:
            return 1

        elif b[0] >= a[1]:
            return -1

        else:
            raise ValueError


    sentence_len = length
    results = []


    spans.sort(key=cmp_to_key(compare))

    idx = -1

    def helper(nest=False):
        nonlocal idx

        idx += 1
        if idx > len(spans) - 1:
            return

        p = spans[idx]
        i, j, label = p

        children = []
        while (
                (idx + 1) < len(spans)
                and i <= spans[idx + 1][0]
                and spans[idx + 1][1] <= j
        ):
            children.append(spans[idx+1])
            helper(True)


        for c in range(len(children)):

            label =  p[2].split('+')[-1]+"<>" if use_fine_grained else 'NULL'

            if c == len(children)-1 and children[-1][1] < p[1]:
                results.append((children[-1][1], p[1], label))

            if c == 0:
                if children[c][0] > p[0]:
                    results.append((p[0], children[c][0], label))

            elif children[c][0] > children[c-1][1]:
                results.append((children[c-1][1], children[c][0], label))

        if nest is False:
            if idx < len(spans) - 1:
                if spans[idx + 1][0] > j:
                    assert ValueError
                helper(False)
        return



    # spans.insert(0, (0, 0, 'START'))
    helper()
    # spans.pop(0)
    spans.extend(results)

    spans.sort(key=cmp_to_key(compare2))

    return spans

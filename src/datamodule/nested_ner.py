
from supar.utils.common import *
from .dm_util.fields import SubwordField, Field, SpanField
import nltk
from fastNLP.core.dataset import DataSet
import logging
from .base import DataModuleBase
from .dm_util.padder import SpanPadder
from .trees import load_trees,  tree2span, get_nongold_span
from functools import cmp_to_key
from  .trees import transition_system

log = logging.getLogger(__name__)


class NestedNERData(DataModuleBase):
    def __init__(self, conf):
        super(NestedNERData, self).__init__(conf)


    def get_inputs(self):
        return ['seq_len', 'chart',  'action_length', 'action_golds', 'span_start', 'span_end', ]

    def get_targets(self):
        return ['raw_annotation', 'raw_raw_word']

    def build_datasets(self):
        datasets = {}
        conf = self.conf
        datasets['train'] = self._load(file=conf.train)
        # if not self.conf.debug:
        datasets['dev'] = self._load(file=conf.dev)
        datasets['test'] = self._load(file=conf.test)
        return datasets

    def _load(self,  file):
        dataset = DataSet()

        sentences = []
        annotations = []
        with open(file, encoding='utf-8') as f:
            for line in f:
                sentences.append(line.strip().split())
                annotations.append(next(f).strip().split('|'))
                next(f)
        length = [len(s) for s in sentences]

        spans = []
        valid = []

        span_start = []
        span_end = []
        label = []
        action_golds = []

        for s, a in zip(sentences, annotations):
            span = process(s, a, True)
            if span is not None:
                spans.append(span)
                valid.append(True)
            else:
                spans.append([])
                valid.append(False)

        for idx, (s, a, l, v) in enumerate(zip(sentences, annotations, length, valid)):
            if not v:
                span_start.append([])
                span_end.append([])
                label.append(['PER'])
                action_golds.append([])

            else:
                s = process(s, a, True)
                ss, se, sl, g =  transition_system(s, l)
                span_start.append(ss)
                span_end.append(se)
                label.append(sl)
                action_golds.append(g)


        dataset.add_field('word', sentences)

        action_length = [len(a) for a in action_golds]

        dataset.add_field('action_length', action_length, )
        dataset.add_field('action_golds', action_golds,)
        dataset.add_field('span_start', span_start)
        dataset.add_field('span_end', span_end)
        dataset.add_field('chart', label)
        dataset.add_field('char', sentences)
        dataset.add_field('raw_word', sentences)
        dataset.add_field('raw_raw_word', sentences)
        dataset.add_field('valid', valid)
        dataset.add_seq_len("raw_word", 'seq_len')
        dataset.add_field("raw_annotation", annotations)

        return dataset

    def _set_padder(self, datasets):
        pass

    def build_fields(self, train_data):
        fields = {}
        fields['word'] = Field('word', pad=PAD, unk=UNK, bos=BOS, eos=EOS, lower=True, min_freq=self.conf.min_freq)
        # fields['pos'] = Field('pos', pad=PAD, unk=UNK, bos=BOS, eos=EOS)
        fields['char'] = SubwordField('char', pad=PAD, unk=UNK, bos=BOS, eos=EOS, subword_eos=subword_eos, subword_bos=subword_bos,
                                      fix_len=self.conf.fix_len)
        # fields['label'] = Field('label', unk=UNK, pad=PAD,)
        # fields[''] = Field('', unk=unk, pad=PAD)
        fields['chart'] = Field('chart', pad=PAD,unk=UNK)
        for name, field in fields.items():
            field.build(train_data[name])
        return fields

    # def _post_process(self, datasets, fields):
    #     vocab = fields.fields['chart'].vocab
    # indexing the training datasets



def process(sentence, annotation, use_fine_grained_null=True):
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

    if annotation[0] == '':
        return [(0, len(sentence), 'NULL')]

    spans = []
    sentence_len = len(sentence)
    results = []

    for idx, a in enumerate(annotation):
        span, label = a.split(" ")
        span = span.split(',')
        spans.append((int(span[0]), int(span[1]), label))

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

            label = (p[2] + '<>') if use_fine_grained_null else 'NULL'

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
                    results.append((j, spans[idx+1][0], 'NULL'))
                helper(False)
            elif idx == (len(spans) -1) and p[1] < sentence_len:
                results.append((p[1], sentence_len, 'NULL'))
        return

    spans.insert(0, (0, 0, '<START>'))
    helper()
    spans.pop(0)
    # if spans[-1][0] != 0:
    spans.append((0, sentence_len, 'NULL'))
    spans.extend(results)
    # 结束咯.
    try:
        spans.sort(key=cmp_to_key(compare2))
    except:
        return None

    return spans




def find_gold_action_target(spans):
    if len(spans) == 0:
        return [], [], []
    focus_word = 0
    target_ptr = []
    target_label = []
    sent_len = spans[-1][1]
    remain_span = []

    for span in spans:
        remain_span.append([focus_word, sent_len])

        if span[0] == focus_word:
            target_ptr.append(span[1])
            focus_word = span[1]

        elif span[1] == focus_word:
            target_ptr.append(span[0])

        else:
            assert ValueError

        target_label.append(span[2])

    return target_ptr, target_label, remain_span




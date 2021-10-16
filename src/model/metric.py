
import logging
import sys
from asyncio import Queue
from collections import Counter
from queue import Empty

import nltk
import subprocess
import torch
from pytorch_lightning.metrics import Metric
from threading import Thread
import regex

from supar.utils.transform import Tree
import tempfile

log = logging.getLogger(__name__)
import os

from pathlib import Path
from functools import cmp_to_key

# copied from Supar
# todo: write result to file.




class NERMetric(Metric):

    def __init__(self, cfg, fields):
        super().__init__()
        self.cfg = cfg
        self.fields = fields
        self.vocab = fields.get_vocab('chart')
        self.add_state("n", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("n_ucm", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("n_multiple", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("c_multiple", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("n_lcm", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("utp", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("ltp", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("gold", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("pred", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.eps = 1e-12
        self.write_result_to_file = cfg.write_result_to_file

        if self.cfg.write_result_to_file:
            self.add_state("outputs", default=[])
            self.prefix = "ner"


    def __call__(self, preds, golds):
        self.update(preds, golds)
        return self


    def update(self, info):

        preds = info['chart_preds']
        golds = info['raw_annotation']

        pr = []
        d = []
        for pred in preds:
            a = []
            b = []
            for p in pred:
                if p[2] == -1:
                    continue
                l = self.vocab[p[2]]
                a.append((p[0], p[1], l))
                if '<>' not in l and l != 'NULL':
                    b.append((p[0], p[1], l))
            pr.append(b)
            d.append(a)
        preds = pr


        if self.cfg.write_result_to_file:
            output = {}
            output['raw_word'] = info['raw_raw_word']
            output['gold_annotation'] = golds
            output['id'] = info['word_id']
            output['preds'] = d
            self.outputs.append(output)


        _n_ucm, _n_lcm, _utp, _ltp, _pred, _gold = 0, 0, 0, 0, 0, 0
        self.n += len(preds)

        _n_multiple = 0
        _c_multiple = 0


        for pred, gold in zip(preds, golds):

            if len(gold) == 1 and gold[0] == '':
                gold = []

            else:
                a = []
                for g in gold:
                    span, label = g.split(' ')
                    span = span.split(',')
                    a.append((int(span[0]), int(span[1]), label))
                gold=a

            upred = Counter([(i, j) for i, j, _ in pred])

            try:
                ugold = Counter([(i, j) for i, j, _ in gold])
            except:
                assert len(gold) == 0
                ugold = Counter()

            utp = list((upred & ugold).elements())
            lpred = Counter(pred)
            lgold = Counter(gold)
            ltp = list((lpred & lgold).elements())
            _n_ucm += float(len(utp) == len(pred) == len(gold))
            _n_lcm += float(len(ltp) == len(pred) == len(gold))
            _utp += len(utp)
            _ltp += len(ltp)
            _pred += len(pred)
            _gold += len(gold)

        self.n_ucm += _n_ucm
        self.n_lcm += _n_lcm
        self.utp += _utp
        self.ltp += _ltp
        self.pred += _pred
        self.gold += _gold

    def compute(self, test=True, epoch_num=-1):
        super(NERMetric, self).compute()
        # if self.cfg.write_result_to_file and epoch_num > 0:
        if self.cfg.write_result_to_file:
            self._write_result_to_file(test=test)
        return self.result

    @property
    def result(self):
        return {
            'c_ucm': self.ucm(),
            'c_lcm': self.lcm(),
            'up': self.up(),
            'ur': self.ur(),
            'uf': self.uf(),
            'lp': self.lp(),
            'lr': self.lr(),
            'lf': self.lf(),
            'score': self.lf(),
        }

    def score(self):
        return self.lf()

    def ucm(self):
        return ( self.n_ucm / (self.n + self.eps)).item()

    def lcm(self):
        return (self.n_lcm / (self.n + self.eps)).item()

    def up(self):
        return (self.utp / (self.pred + self.eps)).item()

    def ur(self):
        return (self.utp / (self.gold + self.eps)).item()

    def uf(self):
        return (2 * self.utp / (self.pred + self.gold + self.eps)).item()

    def lp(self):
        return (self.ltp / (self.pred + self.eps)).item()

    def lr(self):
        return (self.ltp / (self.gold + self.eps)).item()

    def lf(self):
        return (2 * self.ltp / (self.pred + self.gold + self.eps)).item()

    def _write_result_to_file(self, test=False):
        mode = 'test' if test else 'valid'
        outputs = self.outputs

        ids = [output['id'] for output in outputs]
        raw_tree = [output['raw_word'] for output in outputs]
        pred_spans = [output['preds'] for output in outputs]
        gold_spans = [output['gold_annotation'] for output in outputs]

        total_len =  sum(batch.shape[0] for batch in ids)

        final_results = [None for _ in range(total_len)]

        for batch in zip(ids, raw_tree, pred_spans, gold_spans):
            batch_ids, batch_raw_tree, batch_pred_span, batch_gold_span = batch

            for i in range(batch_ids.shape[0]):
                # length = len(batch_word[i])
                # recall that the first token is the imaginary root;
                a = []
                a.append(batch_raw_tree[i])
                a.append(batch_pred_span[i])
                a.append(batch_gold_span[i])
                final_results[batch_ids[i]] = a


        with open(f"{self.prefix}_output_{mode}.txt", 'w', encoding='utf8') as f:
            for (raw_tree, pred_span, gold_span) in final_results:
                f.write(f'{raw_tree}')
                f.write('\n')
                f.write(f'pred_spans:{pred_span}')
                f.write('\n')
                f.write(f'gold_spans:{gold_span}')
                f.write('\n')


class SpanMetric(Metric):
    DELETE = {'TOP', 'S1', '-NONE-', ',', ':', '``', "''", '.', '?', '!', ''}
    EQUAL = {'ADVP': 'PRT'}

    def __init__(self, cfg, fields):
        super().__init__()
        self.cfg = cfg
        self.fields = fields
        self.vocab = fields.get_vocab('chart')
        self.add_state("n", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("n_ucm", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("n_multiple", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("c_multiple", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("n_lcm", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("utp", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("ltp", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("gold", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("pred", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.eps = 1e-12
        self.write_result_to_file = cfg.write_result_to_file
        # why i write this?
        self.transform = True
        if self.cfg.write_result_to_file:
            self.add_state("outputs", default=[])
            self.prefix = "const"



    def __call__(self, preds, golds):
        self.update(preds, golds)
        return self


    def build(self, tree, span):

        leaves = [subtree for subtree in tree.subtrees()
                  if not isinstance(subtree[0], nltk.Tree)]

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

        span.sort(key=cmp_to_key(compare))
        idx = -1

        def helper():
            nonlocal idx
            idx += 1
            i, j, label = span[idx]
            if (i + 1) >= j:
                children = [leaves[i]]
            else:
                children = []
                while (
                        (idx + 1) < len(span)
                        and i <= span[idx + 1][0]
                        and span[idx + 1][1] <= j
                ):
                    children.extend(helper())

            if label:
                if label == 'NULL' or ('<>' in label):
                    return children

                for sublabel in reversed(label.split("+")):
                    children = [nltk.Tree(sublabel, children)]

            return children
        children = helper()
        new =  nltk.Tree("TOP", children)
        assert len(new.pos()) == len(tree.pos())
        return new

    def update(self, info):

        preds = info['chart_preds']
        golds = info['raw_tree']
        golds = [nltk.Tree.fromstring(tree) for tree in golds]

        preds = [ [[i, j, self.vocab[k] if k != -1 else 'NULL'] for i,j,k in tree] for tree in preds]


        preds = [
            self.build(tree, chart)
            for tree, chart in zip(golds, preds)
        ]

        preds = [Tree.factorize(tree, self.DELETE, self.EQUAL) for tree in preds]
        trees = [Tree.factorize(tree, self.DELETE, self.EQUAL) for tree in golds]

        assert len(preds) == len(trees)
        _n_ucm, _n_lcm, _utp, _ltp, _pred, _gold = 0, 0, 0, 0, 0, 0

        self.n += len(preds)

        _n_multiple = 0
        _c_multiple = 0

        if self.cfg.write_result_to_file:
            output = {}
            output['raw_tree'] = info['raw_tree']
            output['gold_spans'] = trees
            output['pred_spans'] = preds
            output['id'] = info['word_id']
            self.outputs.append(output)

        for pred, gold in zip(preds, trees):
            upred = Counter([(i, j) for i, j, _ in pred])
            ugold = Counter([(i, j) for i, j, _ in gold])
            utp = list((upred & ugold).elements())
            lpred = Counter(pred)
            lgold = Counter(gold)
            ltp = list((lpred & lgold).elements())
            _n_ucm += float(len(utp) == len(pred) == len(gold))
            _n_lcm += float(len(ltp) == len(pred) == len(gold))
            _utp += len(utp)
            _ltp += len(ltp)
            _pred += len(pred)
            _gold += len(gold)

        self.n_ucm += _n_ucm
        self.n_lcm += _n_lcm
        self.utp += _utp
        self.ltp += _ltp
        self.pred += _pred
        self.gold += _gold

    def compute(self, test=True, epoch_num=-1):
        super(SpanMetric, self).compute()
        # if self.cfg.write_result_to_file and epoch_num > 0:
        if self.cfg.write_result_to_file:
            self._write_result_to_file(test=test)
        return self.result

    @property
    def result(self):
        return {
            'c_ucm': self.ucm(),
            'c_lcm': self.lcm(),
            'up': self.up(),
            'ur': self.ur(),
            'uf': self.uf(),
            'lp': self.lp(),
            'lr': self.lr(),
            'lf': self.lf(),
            'score': self.lf(),
        }

    def score(self):
        return self.lf()

    def ucm(self):
        return ( self.n_ucm / (self.n + self.eps)).item()

    def lcm(self):
        return (self.n_lcm / (self.n + self.eps)).item()

    def up(self):
        return (self.utp / (self.pred + self.eps)).item()

    def ur(self):
        return (self.utp / (self.gold + self.eps)).item()

    def uf(self):
        return (2 * self.utp / (self.pred + self.gold + self.eps)).item()

    def lp(self):
        return (self.ltp / (self.pred + self.eps)).item()

    def lr(self):
        return (self.ltp / (self.gold + self.eps)).item()

    def lf(self):
        return (2 * self.ltp / (self.pred + self.gold + self.eps)).item()

    def _write_result_to_file(self, test=False):
        mode = 'test' if test else 'valid'
        outputs = self.outputs


        ids = [output['id'] for output in outputs]
        raw_tree = [output['raw_tree'] for output in outputs]
        pred_spans = [output['pred_spans'] for output in outputs]
        gold_spans = [output['gold_spans'] for output in outputs]

        total_len =  sum(batch.shape[0] for batch in ids)

        final_results = [None for _ in range(total_len)]

        for batch in zip(ids, raw_tree, pred_spans, gold_spans):
            batch_ids, batch_raw_tree, batch_pred_span, batch_gold_span = batch
            for i in range(batch_ids.shape[0]):
                # length = len(batch_word[i])
                # recall that the first token is the imaginary root;
                a = []
                a.append(batch_raw_tree[i])
                a.append(batch_pred_span[i])
                a.append(batch_gold_span[i])

                final_results[batch_ids[i]] = a


        with open(f"{self.prefix}_output_{mode}.txt", 'w', encoding='utf8') as f:
            for (raw_tree, pred_span, gold_span) in final_results:
                f.write(raw_tree)
                f.write('\n')
                f.write(f'pred_spans:{pred_span}')
                f.write('\n')
                f.write(f'gold_spans:{gold_span}')
                f.write('\n')



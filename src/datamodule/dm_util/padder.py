from fastNLP.core.field import Padder
import numpy as np

from collections import defaultdict
from ..trees import *

def set_padder(datasets, name, padder):
    for _, dataset in datasets.items():
        dataset.add_field(name, dataset[name].content, padder=padder, ignore_type=True)


class SpanLabelPadder(Padder):
    def __call__(self, contents, field_name, field_ele_dtype, dim: int):
        # max_sent_length = max(rule.shape[0] for rule in contents)
        padded_array = []
        for b_idx, spans in enumerate(contents):
            for (start, end, label) in spans:
                padded_array.append([b_idx, start, end, label])
        return np.array(padded_array)


class SpanPadderVersion2(Padder):
    def __init__(self, vocab):
        super(SpanPadderVersion2, self).__init__()
        self.vocab = vocab
        self.null_idx = self.vocab['NULL']
        self.vocab_size = len(self.vocab)

    def __call__(self, contents, field_name, field_ele_dtype, dim: int):
        # max_sent_length = max(rule.shape[0] for rule in contents)
        parent = defaultdict(list)
        children = defaultdict(list)
        root = []
        transition_count = np.zeros((self.vocab_size, self.vocab_size))

        for b_idx, tree in enumerate(contents):
            def add(i, is_root=False):
                if isinstance(i, InternalParseNode):
                    if is_root:
                        root.append([b_idx, i.left, i.right, self.vocab[i.label]])
                    if i.span_length > 1:
                        parent[i.span_length].append([b_idx, i.left, i.right, self.vocab[i.label]])
                        previous_label = None
                        for child in i.children:
                            if isinstance(child, InternalParseNode):
                                add(child)
                                children[i.span_length].append([b_idx, child.left, child.right, self.vocab[child.label]])
                                if previous_label is not None:
                                    transition_count[previous_label][self.vocab[child.label]] += 1
                                previous_label = self.vocab[child.label]
                            else:
                                children[i.span_length].append([b_idx, child.left, child.right, self.null_idx])
                                if previous_label is not None:
                                    transition_count[previous_label][self.null_idx] += 1
                                previous_label = self.null_idx
                else:
                    pass
            add(tree.convert(), is_root=True)

        a = list(parent.keys())
        a.sort()
        parent2 = []
        child2 = []
        length = []
        for b in a:
            parent2.append(parent[b])
            child2.append(children[b])
            length.append(b)
        return {'transition': transition_count,
                'parent': parent2,
                'children': child2,
                'length': length,
                'root': root}


# for not collapsing labels.
class SpanPadder(Padder):
    def __init__(self, vocab):
        super(SpanPadder, self).__init__()
        self.vocab = vocab
        self.null_idx = self.vocab['NULL']
        self.vocab_size = len(self.vocab)

    def __call__(self, contents, field_name, field_ele_dtype, dim: int):

        parent_span = []
        child_span = []
        root_span = []

        # for the convenience of estimating the loss.
        child_segment_idx = []


        hierarical_span = defaultdict(list)

        for b_idx, tree in enumerate(contents):

            tree = tree.convert()

            def add(node, is_root=False):
                if isinstance(node, InternalParseNode):
                    if node.span_length > 1:
                        parent_span.append([b_idx, node.left, node.right, self.vocab[node.top_label]])
                        if is_root:
                            root_span.append([b_idx, node.left, node.right, self.vocab[node.top_label], len(parent_span) - 1])

                        if len(node.labels) > 1:
                            labels = node.label.split('+')
                            for level in range(len(labels) - 1):
                                hierarical_span[level].append((b_idx, node.left, node.right, self.vocab[labels[level + 1]], len(parent_span)-1))

                        for child in node.children:
                            child_segment_idx.append(len(parent_span)-1)

                            if isinstance(child, InternalParseNode):
                                child_span.append(
                                    [b_idx, child.left, child.right, self.vocab[child.top_label]])
                            else:
                                child_span.append([b_idx, child.left, child.right, self.null_idx])

                        for child in node.children:
                            if isinstance(child, InternalParseNode):
                                add(child)
                else:
                    pass
            add(tree, is_root=True)

        levels = list(hierarical_span.keys())
        levels.sort()
        h_spans = []
        for level in levels:
            h_spans.append(np.array(hierarical_span[level]))

        return {
                'parent_span': np.array(parent_span),
                'child_span': np.array(child_span),
                'root_span': np.array(root_span),
                'hierarical_spans': h_spans,
                'child_segment_idx': np.array(child_segment_idx)
                }




import collections.abc
import gzip
import copy


class TreebankNode(object):
    pass


class InternalTreebankNode(TreebankNode):
    def __init__(self, label, children):
        assert isinstance(label, str)
        self.label = label

        assert isinstance(children, collections.abc.Sequence)
        assert all(isinstance(child, TreebankNode) for child in children)
        assert children
        self.children = tuple(children)

    def linearize(self):
        return "({} {})".format(
            self.label, " ".join(child.linearize() for child in self.children))

    def leaves(self):
        for child in self.children:
            yield from child.leaves()

    def convert(self, index=0, nocache=False):
        tree = self
        sublabels = self.label
        while len(tree.children) == 1 and isinstance(
                tree.children[0], InternalTreebankNode):
            tree = tree.children[0]
            sublabels += f"+{tree.label}"
        children = []
        for child in tree.children:
            children.append(child.convert(index=index))
            index = children[-1].right
        return InternalParseNode(sublabels, children, nocache=nocache)


class LeafTreebankNode(TreebankNode):
    def __init__(self, tag, word):
        assert isinstance(tag, str)
        self.tag = tag

        assert isinstance(word, str)
        self.word = word

    def linearize(self):
        return "({} {})".format(self.tag, self.word)

    def leaves(self):
        yield self

    def convert(self, index=0):
        return LeafParseNode(index, self.tag, self.word)


class ParseNode(object):
    pass


class InternalParseNode(ParseNode):
    def __init__(self, label, children, nocache=False):
        # assert isinstance(label, tuple)
        assert all(isinstance(sublabel, str) for sublabel in label)
        assert label
        self.label = label

        labels = self.label.split("+")

        merged_labels = []

        i = 0

        # to handle NP+NP, VP+VP, for now.
        while i < len(labels):
            if i == len(labels) - 1:
                merged_labels.append(labels[i])
                break
            elif labels[i] == labels[i + 1]:
                merged_labels.append(labels[i] + "+" + labels[i + 1])
                i += 2
            else:
                merged_labels.append(labels[i])
                i += 1

        labels = merged_labels
        self.top_label = labels[0]
        self.labels = labels

        assert isinstance(children, collections.abc.Sequence)
        assert all(isinstance(child, ParseNode) for child in children)
        assert children
        assert len(children) > 1 or isinstance(children[0], LeafParseNode)
        assert all(
            left.right == right.left
            for left, right in zip(children, children[1:]))
        self.children = tuple(children)
        self.left = children[0].left
        self.right = children[-1].right
        self.span_length = self.right - self.left
        self.nocache = nocache

    def leaves(self):
        for child in self.children:
            yield from child.leaves()

    def convert(self):
        children = [child.convert() for child in self.children]
        tree = InternalTreebankNode(self.label[-1], children)
        for sublabel in reversed(self.label[:-1]):
            tree = InternalTreebankNode(sublabel, [tree])
        return tree

    def enclosing(self, left, right):
        assert self.left <= left < right <= self.right
        for child in self.children:
            if isinstance(child, LeafParseNode):
                continue
            if child.left <= left < right <= child.right:
                return child.enclosing(left, right)
        return self

    def oracle_label(self, left, right):
        enclosing = self.enclosing(left, right)
        if enclosing.left == left and enclosing.right == right:
            return enclosing.label
        return ()

    def oracle_splits(self, left, right):
        return [
            child.left
            for child in self.enclosing(left, right).children
            if left < child.left < right
        ]


class LeafParseNode(ParseNode):
    def __init__(self, index, tag, word):
        assert isinstance(index, int)
        assert index >= 0
        self.left = index
        self.right = index + 1

        assert isinstance(tag, str)
        self.tag = tag

        assert isinstance(word, str)
        self.word = word

    def leaves(self):
        yield self

    def convert(self):
        return LeafTreebankNode(self.tag, self.word)


def tree_from_str(treebank, strip_top=True, strip_spmrl_features=True):
    # Features bounded by `##` may contain spaces, so if we strip the features
    # we need to do so prior to tokenization
    if strip_spmrl_features:
        treebank = "".join(treebank.split("##")[::2])

    tokens = treebank.replace("(", " ( ").replace(")", " ) ").split()

    def helper(index):
        trees = []

        while index < len(tokens) and tokens[index] == "(":
            paren_count = 0
            while tokens[index] == "(":
                index += 1
                paren_count += 1

            label = tokens[index]
            index += 1

            if tokens[index] == "(":
                children, index = helper(index)
                trees.append(InternalTreebankNode(label, children))
            else:
                word = tokens[index]
                index += 1
                trees.append(LeafTreebankNode(label, word))

            while paren_count > 0:
                assert tokens[index] == ")"
                index += 1
                paren_count -= 1

        return trees, index

    trees, index = helper(0)
    assert index == len(tokens)

    if strip_top:
        for i, tree in enumerate(trees):
            if tree.label in ("TOP", "ROOT"):
                assert len(tree.children) == 1
                trees[i] = tree.children[0]

    assert len(trees) == 1

    return trees[0]


def load_trees(path, strip_top=True, strip_spmrl_features=True):
    with open(path) as infile:
        treebank = infile.read()

    # Features bounded by `##` may contain spaces, so if we strip the features
    # we need to do so prior to tokenization
    if strip_spmrl_features:
        treebank = "".join(treebank.split("##")[::2])

    tokens = treebank.replace("(", " ( ").replace(")", " ) ").split()

    def helper(index):
        trees = []

        while index < len(tokens) and tokens[index] == "(":
            paren_count = 0
            while tokens[index] == "(":
                index += 1
                paren_count += 1

            label = tokens[index]
            index += 1

            if tokens[index] == "(":
                children, index = helper(index)
                trees.append(InternalTreebankNode(label, children))
            else:
                word = tokens[index]
                index += 1
                trees.append(LeafTreebankNode(label, word))

            while paren_count > 0:
                assert tokens[index] == ")"
                index += 1
                paren_count -= 1

        return trees, index

    trees, index = helper(0)
    assert index == len(tokens)

    if strip_top:
        for i, tree in enumerate(trees):
            if tree.label in ("TOP", "ROOT"):
                assert len(tree.children) == 1
                trees[i] = tree.children[0]
    return trees


def tree2span(tree, collapse=False):
    span = []

    def add(i):
        if isinstance(i, InternalParseNode):
            if collapse:
                span.append((i.left, i.right, i.label))
            else:
                labels = i.labels
                for label in labels:
                    span.append((i.left, i.right, label))

            if i.span_length > 1:
                for child in i.children:
                    add(child)
        else:
            span.append((i.left, i.right, 'NULL'))

    add(tree)
    return span


import numpy as np


def get_nongold_span(tree, span):
    tree = tree.convert()
    span_length = tree.span_length
    oracle_lookup = [[[0, 0] for _ in range(span_length + 1)] for _ in range(span_length + 1)]

    # constructing the array.
    for s in span:
        start, end, label = s
        # label = self.vocab(label)
        pair = [end, label]
        oracle_lookup[start][end] = pair
        i = end + 1
        while i < span_length:
            if oracle_lookup[start][i][0] < end:
                oracle_lookup[start][i] = pair
                i += 1
            else:
                break

    oracle_parent_child_span = []

    # add oracles.
    for i in range(span_length):
        for j in range(i + 1, span_length):
            # meaning this span does not exist
            if oracle_lookup[i][j][0] != j:
                # parent span does not need label..? we want to normalize all of it. right.
                # oracle_parent_span.append([b_idx, i, j])
                parent_span = [i, j]
                child_span = []
                # start idx?
                k = i
                # parent and children co-normalization.
                while k != j:
                    child_span.append([k, oracle_lookup[k][j][0], oracle_lookup[k][j][1]])
                    k = oracle_lookup[k][j][0]
                oracle_parent_child_span.append([parent_span, child_span])

    return oracle_parent_child_span


def transition_system(span, length):
    # start symbol (0, 0), end symbol (length, length)
    max_branching = 0
    max_span_len = 0

    start = [(0, 0, '<START>')]
    remain = [(0, length)]

    focus_word = 0

    stack = []
    masks = []
    golds = []


    for s in span:

        if s[0] > focus_word:
            raise ValueError

        elif s[1] > focus_word and s[0] != focus_word:
            raise ValueError

        mask = [1 for _ in range(length + 1)]
        mask[0] = 0
        if len(stack) > 0:
            for j in range(stack[-1][1] + 1):
                mask[j] = 0

        for element in stack[:-1]:
            mask[element[0]] = 1
        masks.append(mask)

        if len(stack) > 0:
            if stack[-1][1] == s[0]:
                stack.append((s[0], s[1]))
                max_span_len = max(s[1] - s[0] + 1, max_span_len)
                golds.append(s[1])
            else:
                i = 0
                while stack[-1][0] != s[0]:
                    stack.pop()
                    i+=1
                stack.pop()
                i+=1
                max_branching = max(max_branching, i)
                stack.append((s[0], s[1]))
                golds.append(s[0])
        else:
            assert s[0] == 0
            golds.append(s[1])
            stack.append((s[0], s[1]))

        focus_word = max(s[1], focus_word)
        if s[0] == 0 and s[1] == length:
            break

        start.append((s[0], s[1], s[2]))
        remain.append((focus_word, length))


    assert len(start) == len(remain)
    assert len(golds) == len(start)

    output = copy.deepcopy(start)
    output.pop(0)
    output.append(span[-1])

    span_start = [x[0] for x in output]
    span_end = [x[1] for x in output]
    span_label = [x[2] for x in output]

    return span_start, span_end, span_label, golds










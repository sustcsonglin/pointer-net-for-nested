# -*- coding: utf-8 -*-

from collections import namedtuple

import torch
import torch.distributed as dist
from supar.utils.alg import kmeans


class Dataset(torch.utils.data.Dataset):
    r"""
    Dataset that is compatible with :class:`torch.utils.data.Dataset`.
    This serves as a wrapper for manipulating all data fields
    with the operating behaviours defined in :class:`Transform`.
    The data fields of all the instantiated sentences can be accessed as an attribute of the dataset.

    Args:
        transform (Transform):
            An instance of :class:`Transform` and its derivations.
            The instance holds a series of loading and processing behaviours with regard to the specfic data format.
        data (list[list] or str):
            A list of instances or a filename.
            This will be passed into :meth:`transform.load`.
        kwargs (dict):
            Keyword arguments that will be passed into :meth:`transform.load` together with `data`
            to control the loading behaviour.

    Attributes:
        transform (Transform):
            An instance of :class:`Transform`.
        sentences (list[Sentence]):
            A list of sentences loaded from the data.
            Each sentence includes fields obeying the data format defined in ``transform``.
    """

    def __init__(self, transform, data, **kwargs):
        super(Dataset, self).__init__()
        self.transform = transform
        self.sentences = transform.load(data, **kwargs)

    def __repr__(self):
        s = f"{self.__class__.__name__}("
        s += f"n_sentences={len(self.sentences)}"
        if hasattr(self, 'loader'):
            s += f", n_batches={len(self.loader)}"
        if hasattr(self, 'buckets'):
            s += f", n_buckets={len(self.buckets)}"
        s += ")"
        return s

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        if not hasattr(self, 'fields'):
            raise RuntimeError("The fields are not numericalized. Please build the dataset first.")
        for d in self.fields.values():
            yield d[index]

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        return [getattr(sentence, name) for sentence in self.sentences]

    def __setattr__(self, name, value):
        if 'sentences' in self.__dict__ and name in self.sentences[0]:
            # restore the order of sequences in the buckets
            indices = torch.tensor([i
                                    for bucket in self.buckets.values()
                                    for i in bucket]).argsort()
            for index, sentence in zip(indices, self.sentences):
                setattr(sentence, name, value[index])
        else:
            self.__dict__[name] = value

    def __getstate__(self):
        # only pickle the Transform object and sentences
        return {'transform': self.transform, 'sentences': self.sentences}

    def __setstate__(self, state):
        self.__dict__.update(state)

    def collate_fn(self, batch):
        return {f: d for f, d in zip(self.fields.keys(), zip(*batch))}

    def build(self, batch_size, n_buckets=1, shuffle=False, distributed=False):
        # numericalize all fields
        self.fields = self.transform(self.sentences)
        # NOTE: the final bucket count is roughly equal to n_buckets
        self.lengths = [len(i) for i in self.fields[next(iter(self.fields))]]
        self.buckets = dict(zip(*kmeans(self.lengths, n_buckets)))
        self.loader = DataLoader(dataset=self,
                                 batch_sampler=Sampler(buckets=self.buckets,
                                                       batch_size=batch_size,
                                                       shuffle=shuffle,
                                                       distributed=distributed),
                                 collate_fn=self.collate_fn)


class JointDataset(torch.utils.data.Dataset):
    def __init__(self, fields):
        super(JointDataset, self).__init__()
        self.fields = fields

    def collate_fn(self, batch):
        return {f: d for f, d in zip(self.fields.keys(), zip(*batch))}

    def __getitem__(self, index):
       for d in self.fields.values():
            yield d[index]

    def build(self, batch_size, n_buckets=1, shuffle=False, distributed=False):
        # numericalize all fields
        # self.fields = self.transform(self.sentences)
        # NOTE: the final bucket count is roughly equal to n_buckets
        self.lengths = [len(i) for i in self.fields[next(iter(self.fields))]]
        self.buckets = dict(zip(*kmeans(self.lengths, n_buckets)))
        self.loader = DataLoader(dataset=self,
                                 batch_sampler=Sampler(buckets=self.buckets,
                                                       batch_size=batch_size,
                                                       shuffle=shuffle,
                                                       distributed=distributed),
                                 collate_fn=self.collate_fn)




class DataLoader(torch.utils.data.DataLoader):
    r"""
    DataLoader, matching with :class:`Dataset`.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __iter__(self):
        for batch in super().__iter__():
            yield namedtuple('Batch', [f.name for f in batch.keys()])(*[f.compose(d) for f, d in batch.items()])

class Sampler(torch.utils.data.Sampler):
    r"""
    Sampler that supports for bucketization and token-level batchification.

    Args:
        buckets (dict):
            A dict that maps each centroid to indices of clustered sentences.
            The centroid corresponds to the average length of all sentences in the bucket.
        batch_size (int):
            Token-level batch size. The resulting batch contains roughly the same number of tokens as ``batch_size``.
        shuffle (bool):
            If ``True``, the sampler will shuffle both buckets and samples in each bucket. Default: ``False``.
        distributed (bool):
            If ``True``, the sampler will be used in conjunction with :class:`torch.nn.parallel.DistributedDataParallel`
            that restricts data loading to a subset of the dataset.
            Default: ``False``.
    """

    def __init__(self, buckets, batch_size, shuffle=False, distributed=False, evaluate=False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sizes, self.buckets = zip(*[(size, bucket) for size, bucket in buckets.items()])
        # number of chunks in each bucket, clipped by range [1, len(bucket)]
        self.chunks = [min(len(bucket), max(round(size * len(bucket) / batch_size), 1))
                       for size, bucket in zip(self.sizes, self.buckets)]
        self.rank = dist.get_rank() if distributed else 0
        self.rank = dist.get_rank() if distributed else 0
        self.replicas = dist.get_world_size() if distributed else 1
        self.force_even = distributed and evaluate
        print(self.force_even)
        self.samples = sum(self.chunks) // self.replicas +  ( self.replicas *  int( sum(self.chunks) % self.replicas) if self.force_even else 0)
        self.epoch = 0

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)
        range_fn = torch.arange
        # if `shuffle=True`, shuffle both the buckets and samples in each bucket
        # for distributed training, make sure each process genertes the same random sequence at each epoch
        if self.shuffle:
            def range_fn(x):
                return torch.randperm(x, generator=g)

        total, count = 0, 0
        # TODO: more elegant way to deal with uneven data, which we directly discard right now

        if self.force_even:
            all = []
            for i in range_fn(len(self.buckets)).tolist():
                split_sizes = [(len(self.buckets[i]) - j - 1) // self.chunks[i] + 1
                               for j in range(self.chunks[i])]
                # DON'T use `torch.chunk` which may return wrong number of chunk
                for batch in range_fn(len(self.buckets[i])).split(split_sizes):
                    all.append([self.buckets[i][j] for j in batch.tolist()])
            if len(all) % self.replicas != 0:
                for batch in all:
                    if len(batch) < self.replicas:
                        continue
                    all.remove(batch)
                    l = self.replicas - ( (len(all)+1) % self.replicas) + 1
                    for i in range(l):
                        if i < l-1:
                            all.append([batch[i]])
                        else:
                            all.append(batch[l-1:])
                    break
            length = 0
            assert len(all) % self.replicas == 0, f"{len(all)}, {self.replicas}"
            for i, batch in enumerate(all):
                if i % self.replicas == self.rank:
                    length += len(batch)
                    yield batch

        else:
            for i in range_fn(len(self.buckets)).tolist():
                split_sizes = [(len(self.buckets[i]) - j - 1) // self.chunks[i] + 1
                               for j in range(self.chunks[i])]

                if self.force_even:
                    if len(split_sizes) % self.replicas != 0:
                        top = split_sizes.pop()

                        v = top // (self.replicas + 1)

                        while top != 0:
                            split_sizes.append(min(v, top))
                            top -= min(v, top)

                # DON'T use `torch.chunk` which may return wrong number of chunks
                for batch in range_fn(len(self.buckets[i])).split(split_sizes):
                    if count == self.samples:
                        break
                    if total % self.replicas == self.rank:
                        count += 1
                        yield [self.buckets[i][j] for j in batch.tolist()]
                    total += 1

        self.epoch += 1

    def __len__(self):
        return self.samples
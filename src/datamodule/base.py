import pytorch_lightning as pl
import os
from supar.utils.common import *
import pickle
from .dm_util.fields import SubwordField, Field
import hydra
import nltk
from fastNLP.core.dataset import DataSet
from fastNLP.core.sampler import RandomSampler
import torch
from supar.utils import Embedding
from .dm_util.datamodule_util import get_sampler
from fastNLP.core.batch import DataSetIter
import logging
from abc import ABC
import copy
log = logging.getLogger(__name__)
import numpy as np
from src.model.module.ember.ext_embedding import ExternalEmbeddingSupar




class Fields():
    def __init__(self, fields, inputs, config):
        self.fields = fields
        self.inputs = self._get_true_input_fields(inputs)
        # datamodule config..
        self.conf = config
        self.root_dir = os.getcwd()

    def get_bert_name(self):
        if 'bert' in self.fields.keys():
            return self.fields['bert_name']
        else:
            raise ValueError

    def get_ext_emb(self):
        if 'ext_emb' in self.fields:
            return self.fields['ext_emb']
        else:
            return None

    def _get_true_input_fields(self, inputs):
        true_inputs = []
        for i in inputs:
            if i in self.fields:
                true_inputs.append(i)
        return true_inputs

    def get_vocab_size(self, name):
            return len(self.fields[name].vocab)

    def get_name_from_id(self, name, id):
        return self.fields[name].vocab[id]

    def get_pad_index(self, name):
        return self.fields[name].pad_index

    def get_vocab(self, name):
        return self.fields[name].vocab


class DataModuleBase(pl.LightningDataModule):
    def __init__(self, conf):
        super(DataModuleBase, self).__init__()
        self.conf = conf

    def build_datasets(self):
        raise NotImplementedError

    def build_fields(self, train_data):
        raise NotImplementedError

    def get_inputs(self):
        raise NotImplementedError

    def get_targets(self):
        raise NotImplementedError

    def _set_padd_er(self, datasets):
        pass

    def _make_assertion(self,datasets):
        pass

    def _post_process(self, datasets, fields):
        pass



    def setup(self):
        datasets, fields = self._build_dataset_and_fields()
        self._make_assertion(datasets)
        inputs = self.get_inputs()
        targets = self.get_targets()

        # transform sequence of word to sequences of sequence of subword idx.
        if self.conf.use_bert:
            self._add_bert_to_field(datasets, fields)
            inputs.append('bert')

        # extend the vocabulary, reindex the word field; copied from Supar.
        if self.conf.use_emb and self.conf.use_word:
            self._build_external_emb(datasets, fields)

        if self.conf.use_char:
            inputs.append("char")

        if self.conf.use_word:
            inputs.append("word")

        if self.conf.use_pos:
            inputs.append('pos')

        # add word id
        for dataset in datasets.values():
            dataset.add_field('word_id', [i for i in range(len(dataset))])

        inputs.append('word_id')
        set_input(datasets, inputs)
        set_target(datasets, targets)

        try:
            valid = np.array(datasets['train']["valid"])
            datasets['train'] = datasets['train'].drop(lambda x: x['valid'] == False, inplace=True)
            log.info(
                f"Dataset Loaded. Total training sentences:{valid.shape[0]}, Total valid training sentences:{valid.sum()}")

        except:
            log.info(
                f"Dataset Loaded. Total training sentences:{len(datasets['train']['word'])}, Do not filter out invalid sentences.")

        log.info(f"max_len:{self.conf.max_len}, before_drop: {len(datasets['train']['word'])}")
        datasets['train'] = datasets['train'].drop(lambda x: x['seq_len'] > self.conf.max_len, inplace=True)
        datasets['train'] = datasets['train'].drop(lambda x: x['seq_len'] < 2, inplace=True)
        log.info(f"after drop: {len(datasets['train']['word'])}")

        log.info(f"max_len:{self.conf.max_len}, before_drop: dev: {len(datasets['dev']['word'])}, test:{len(datasets['test']['word'])}")
        datasets['dev'] = datasets['dev'].drop(lambda x: x['seq_len'] > self.conf.max_len_test, inplace=True)
        datasets['test'] = datasets['test'].drop(lambda x: x['seq_len'] > self.conf.max_len_test, inplace=True)
        log.info(f"after drop: dev: {len(datasets['dev']['word'])}, test:{len(datasets['test']['word'])}")

        log.info(f"Train: {len(datasets['train']['word'])} sentences, valid: {len(datasets['dev']['word'])} sentences, test: {len(datasets['test']['word'])} sentences")
        log.info(f"Training max tokens: {self.conf.max_tokens}, total_bucket:{self.conf.bucket}")
        log.info(f"Testing max tokens: {self.conf.max_tokens_test}, total_bucket:{self.conf.bucket_test}")
        log.info(f"input: {inputs}")

        self.datasets = datasets
        self.fields = Fields(fields=fields, inputs=inputs, config=self.conf)
        self._post_process(datasets, fields)
        self._set_padder(datasets)


    def _index_datasets(self, fields, datasets):
        for _, dataset in datasets.items():
            for name, field in fields.items():
                dataset.apply_field(func=field.transform, field_name=name, new_field_name=name)

    # Supar's implementation for now.
    def _build_external_emb(self, datasets, fields):
        assert self.conf.ext_emb_path, ("The external word embedding path does not exsit, please check.")
        log.info(f"use external embeddings :{self.conf.ext_emb_path}")
        has_cache = False
        cache_path = self.conf.ext_emb_path + ".cache.pickle"
        if self.conf.use_cache and  os.path.exists(cache_path):
            log.info(f"Load cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                cache = pickle.load(f)
            WORD = cache['word']
            has_cache = True

        # https://github.com/yzhangcs/parser/blob/main/supar/utils/field.py line178
        if not has_cache:
            log.info("Find no cache, building..")
            cache = {}
            if 'glove' in self.conf.ext_emb_path:
                unk = 'unk'
            else:
                unk = None

            embed = Embedding.load(self.conf.ext_emb_path, unk=unk)
            WORD = fields['word']

            tokens = WORD.preprocess(embed.tokens)
            if embed.unk:
                tokens[embed.unk_index] = WORD.unk


            cache['origin_len'] = len(WORD.vocab)

            WORD.vocab.extend(tokens)
            embedding = torch.zeros(len(WORD.vocab), embed.dim)
            embedding[WORD.vocab[tokens]] = embed.vectors
            embedding /= torch.std(embedding)
            cache['embed'] = embedding

            for name, d in datasets.items():
                cache[name] = [WORD.transform(instance) for instance in d.get_field('raw_word').content]

            cache['word'] = WORD

            self._dump(cache_path=cache_path, to_cache=cache)

        for name, d in datasets.items():
            d.add_field('word', cache[name])

        fields['word'] = WORD
        fields['ext_emb'] = ExternalEmbeddingSupar(cache['embed'], cache['origin_len'], WORD.unk_index)
        log.info(f"before extend, vocab_size:{cache['origin_len']}")
        log.info(f"extended_vocab_size:{cache['embed'].shape[0]}")


    def _build_dataset_and_fields(self):
        log.info(f"looking for cache:{self.conf.cache}, use_cache:{self.conf.use_cache}")
        if os.path.exists(self.conf.cache) and self.conf.use_cache:
            with open(self.conf.cache, 'rb') as f:
                cache = pickle.load(f)
                datasets = cache['datasets']
                fields = cache['fields']
            log.info(f"load cache:{self.conf.cache}")
        else:
            log.info("creating dataset.")
            # return: {"train", "dev","test"}
            datasets = self.build_datasets()
            fields = self.build_fields(datasets['train'])
            self._index_datasets(fields, datasets)
            self._dump(cache_path=self.conf.cache, to_cache={'datasets': datasets,
                         'fields': fields})
        return datasets, fields


    def _add_bert_to_field(self, datasets, fields):

        log.info(f"Use bert:{self.conf.bert}")
        if not os.path.exists(self.conf.cache_bert) or not self.conf.use_bert_cache:
            BERT = get_bert(self.conf.bert, fix_len=self.conf.fix_len)
            def get_bert_cache(datasets):
                cache_bert = {}
                cache_bert['bert'] = BERT
                for name, d in datasets.items():
                    cache_bert[name] = [BERT.transform(instance) for instance in d.get_field('raw_raw_word').content]
                return cache_bert
            cache_bert = get_bert_cache(datasets)
            self._dump(self.conf.cache_bert, cache_bert)
        else:
            log.info(f"load cache bert:{self.conf.cache_bert}")
            with open(self.conf.cache_bert, 'rb') as f:
                cache_bert = pickle.load(f)
            BERT = cache_bert['bert']

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.conf.bert)
        pad_id = tokenizer.pad_token_id

        # important
        for name, d in datasets.items():
            d.add_field('bert', cache_bert[name], )
            d.set_pad_val('bert', pad_id)

        fields['bert'] = BERT
        fields['bert_name'] = self.conf.bert


    def train_dataloader(self):
        if self.conf.train_sampler_type == 'token':
            length = self.datasets['train'].get_field('seq_len').content
            sampler = get_sampler(lengths=length, max_tokens=self.conf.max_tokens,
                                n_buckets=self.conf.bucket, distributed=self.conf.distributed, evaluate=False)
            return DataSetIter(self.datasets['train'], batch_size=1, sampler=None, as_numpy=False, num_workers=4,
                               pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None,
                               batch_sampler=sampler)
        else:
            sampler = RandomSampler()
            return DataSetIter(self.datasets['train'], sampler=sampler, batch_size=self.conf.batch_size, max_tokens=self.conf.max_tokens, num_workers=4)

    def val_dataloader(self):
        if self.conf.test_sampler_type == 'token':
            length = self.datasets['dev'].get_field('seq_len').content
            sampler = get_sampler(lengths=length, max_tokens=self.conf.max_tokens_test,
                                n_buckets=self.conf.bucket_test, distributed=self.conf.distributed, evaluate=True)
            val_loader = DataSetIter(self.datasets['dev'], batch_size=1, sampler=None, as_numpy=False, num_workers=4,
                               pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None,
                               batch_sampler=sampler)
            return val_loader
        else:
            sampler = RandomSampler()
            return DataSetIter(self.datasets['dev'], sampler=sampler, batch_size=self.conf.batch_size, max_tokens=self.conf.max_tokens_test)

    def test_dataloader(self):
        if self.conf.test_sampler_type == 'token':
            length = self.datasets['test'].get_field('seq_len').content
            sampler = get_sampler(lengths=length, max_tokens=self.conf.max_tokens_test,
                                n_buckets=self.conf.bucket_test, distributed=self.conf.distributed, evaluate=True)
            return DataSetIter(self.datasets['test'], batch_size=1, sampler=None, as_numpy=False, num_workers=4,
                               pin_memory=False,drop_last=False, timeout=0, worker_init_fn=None,
                               batch_sampler=sampler)
        else:
            sampler = RandomSampler()
            return DataSetIter(self.datasets['test'], sampler=sampler, batch_size=self.conf.batch_size, max_tokens=self.conf.max_tokens_test)

    def _dump(self, cache_path, to_cache):
            with open(cache_path, 'wb') as f:
                pickle.dump(to_cache, f)



def set_input(datasets, inputs):
    for i in inputs:
        for dataset in datasets.values():
            dataset.set_input(i)

def set_target(datasets, targets):
    for t in targets:
        for dataset in datasets.values():
            dataset.set_target(t)

def isProjective(heads):
    pairs = [(h, d) for d, h in enumerate(heads, 1) if h >= 0]
    for i, (hi, di) in enumerate(pairs):
        for hj, dj in pairs[i+1:]:
            (li, ri), (lj, rj) = sorted([hi, di]), sorted([hj, dj])
            if li <= hj <= ri and hi == dj:
                return False
            if lj <= hi <= rj and hj == di:
                return False
            if (li < lj < ri or li < rj < ri) and (li - lj)*(ri - rj) > 0:
                return False
    return True


def get_bert(bert_name, fix_len=20):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(bert_name)
    BERT = SubwordField(bert_name,
                        pad=tokenizer.pad_token,
                        unk=tokenizer.unk_token,
                        bos=tokenizer.cls_token or tokenizer.cls_token,
                        eos=tokenizer.sep_token or tokenizer.sep_token,
                        fix_len=fix_len,
                        tokenize=tokenizer.tokenize)
    BERT.vocab = tokenizer.get_vocab()
    return BERT









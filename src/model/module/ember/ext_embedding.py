import torch.nn as nn
import logging


log = logging.getLogger(__name__)

# As Supar.
class ExternalEmbeddingSupar(nn.Module):
    def __init__(self, emb, origin_word_size, unk_index):
        super(ExternalEmbeddingSupar, self).__init__()

        self.pretrained = nn.Embedding.from_pretrained(emb)
        self.origin_word_size = origin_word_size
        self.word_emb = nn.Embedding(origin_word_size, emb.shape[-1])
        self.unk_index = unk_index
        nn.init.zeros_(self.word_emb.weight)

    def forward(self, words):
        ext_mask = words.ge(self.word_emb.num_embeddings)
        ext_words = words.masked_fill(ext_mask, self.unk_index)
        # get outputs from embedding layers
        word_embed = self.word_emb(ext_words)
        word_embed += self.pretrained(words)
        return word_embed

    def get_dim(self):
        return self.word_emb.weight.shape[-1]
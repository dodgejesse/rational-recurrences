import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def deep_iter(x):
    if isinstance(x, list) or isinstance(x, tuple):
        for u in x:
            for v in deep_iter(u):
                yield v
    else:
        yield x

class CNN_Text(nn.Module):

    def __init__(self, n_in, widths=[3,4,5], filters=100):
        super(CNN_Text,self).__init__()
        Ci = 1
        Co = filters
        h = n_in
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (w, h)) for w in widths])

    def forward(self, x):
        # x is (batch, len, d)
        x = x.unsqueeze(1) # (batch, Ci, len, d)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1] #[(batch, Co, len), ...]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] #[(N,Co), ...]
        x = torch.cat(x, 1)
        return x


class EmbeddingLayer(nn.Module):
    def __init__(self, words, embs=None, fix_emb=True, sos='<s>', eos='</s>',
                 oov='<oov>', pad='<pad>', normalize=True, bert_embed=False):
        super(EmbeddingLayer, self).__init__()
        word2id = {}

        self.bert_embed = bert_embed
        if bert_embed:
            self.word2id = None
            self.emb_size = len(words[0][0])
            return
        
        if embs is not None:
            embwords, embvecs = embs
            for word in embwords:
                assert word not in word2id, "Duplicate words in pre-trained embeddings"
                word2id[word] = len(word2id)
            emb_size = len(embvecs[0])
            sys.stdout.write("{} pre-trained word embeddings with dim={} loaded.\n".format(len(word2id),
                                                                                           emb_size))
        for w in deep_iter(words):
            if w not in word2id:
                word2id[w] = len(word2id)

        if oov not in word2id:
            word2id[oov] = len(word2id)
        if pad not in word2id:
            word2id[pad] = len(word2id)
        if sos not in word2id:
            word2id[sos] = len(word2id)
        if eos not in word2id:
            word2id[eos] = len(word2id)
        self.word2id = word2id
        self.n_V, self.emb_size = len(word2id), emb_size
        self.oovid = word2id[oov]
        self.padid = word2id[pad]
        self.sosid = word2id[sos]
        self.eosid = word2id[eos]
        self.embedding = nn.Embedding(self.n_V, emb_size)
        self.embedding.weight.data.uniform_(-0.25, 0.25)

        if embs is not None:
            weight  = self.embedding.weight
            weight.data[:len(embwords)].copy_(torch.from_numpy(embvecs))
            sys.stdout.write("embedding shape: {}\n".format(weight.size()))
        if normalize:
            weight = self.embedding.weight
            norms = weight.data.norm(2,1)
            if norms.dim() == 1:
                norms = norms.unsqueeze(1)
            weight.data.div_(norms.expand_as(weight.data))
        pad_vec = np.zeros((1, emb_size), dtype=np.float32)
        self.embedding.weight.data[self.padid:self.padid+1,:].copy_(torch.from_numpy(pad_vec))
        if fix_emb:
            self.embedding.weight.requires_grad = False

    def forward(self, input):
        if self.bert_embed:
            return input
        else:
            return self.embedding(input)

import gzip
import os
import sys
import re
import random
import json

import numpy as np
import torch

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()

def read_corpus(path, clean=True, TREC=False, encoding='ISO-8859-1'):
    data = []
    labels = []
    with open(path, encoding=encoding) as fin:
        for line in fin:
            label, sep, text = line.partition('\t')
            label = int(label)
            text = clean_str(text.strip()) if clean else text.strip()
            labels.append(label)
            data.append(text.split())
    return data, labels

def read_MR(path, seed=1234):
    file_path = os.path.join(path, "rt-polarity.all")
    data, labels = read_corpus(file_path, encoding='latin-1')
    random.seed(seed)
    perm = list(range(len(data)))
    random.shuffle(perm)
    data = [ data[i] for i in perm ]
    labels = [ labels[i] for i in perm ]
    return data, labels

def read_SUBJ(path, seed=1234):
    file_path = os.path.join(path, "subj.all")
    data, labels = read_corpus(file_path)
    random.seed(seed)
    perm = list(range(len(data)))
    random.shuffle(perm)
    data = [ data[i] for i in perm ]
    labels = [ labels[i] for i in perm ]
    return data, labels

def read_CR(path, seed=1234):
    file_path = os.path.join(path, "custrev.all")
    data, labels = read_corpus(file_path)
    random.seed(seed)
    perm = list(range(len(data)))
    random.shuffle(perm)
    data = [ data[i] for i in perm ]
    labels = [ labels[i] for i in perm ]
    return data, labels

def read_MPQA(path, seed=1234):
    file_path = os.path.join(path, "mpqa.all")
    data, labels = read_corpus(file_path)
    random.seed(seed)
    perm = list(range(len(data)))
    random.shuffle(perm)
    data = [ data[i] for i in perm ]
    labels = [ labels[i] for i in perm ]
    return data, labels

def read_TREC(path, seed=1234):
    train_path = os.path.join(path, "TREC.train.all")
    test_path = os.path.join(path, "TREC.test.all")
    train_x, train_y = read_corpus(train_path, TREC=True)
    test_x, test_y = read_corpus(test_path, TREC=True)
    random.seed(seed)
    perm = list(range(len(train_x)))
    random.shuffle(perm)
    train_x = [ train_x[i] for i in perm ]
    train_y = [ train_y[i] for i in perm ]
    return train_x, train_y, test_x, test_y

def read_SST(path, seed=1234):
    train_path = os.path.join(path, "train")
    valid_path = os.path.join(path, "dev")
    test_path = os.path.join(path, "test")
    train_x, train_y = read_corpus(train_path, clean=True)
    valid_x, valid_y = read_corpus(valid_path, clean=True)
    test_x, test_y = read_corpus(test_path, clean=True)
    random.seed(seed)
    perm = list(range(len(train_x)))
    random.shuffle(perm)
    train_x = [ train_x[i] for i in perm ]
    train_y = [ train_y[i] for i in perm ]
    return train_x, train_y, valid_x, valid_y, test_x, test_y

# to read a dataset preprocessed with bert embeddings.
def read_bert(path, read_train=True, seed=1234):
    valid_path = os.path.join(path, "dev_bert")
    valid_x, valid_y = read_bert_file(valid_path)


    if read_train:
        train_path = os.path.join(path, "train_bert")
        train_x, train_y = read_bert_file(train_path)
        random.seed(seed)
        perm = list(range(len(train_x)))
        random.shuffle(perm)
        train_x = [ train_x[i] for i in perm ]
        train_y = [ train_y[i] for i in perm ]

        return train_x, train_y, valid_x, valid_y, [], []#, test_x, test_y
    else:
        test_path = os.path.join(path, "test_bert")
        test_x, test_y = read_bert_file(test_path)

        [], [], valid_x, valid_y, test_x, test_y

def read_bert_file(path):
    data = []
    labels = []
    with open(path) as fin:
        for line in fin:
            label, embeds_string = line.split('\t')
            label = int(label)
            embeds = json.loads(embeds_string)
            labels.append(label)
            data.append(embeds)
    return data, labels

def cv_split(data, labels, nfold, test_id):
    assert (nfold > 1) and (test_id >= 0) and (test_id < nfold)
    lst_x = [ x for i, x in enumerate(data) if i%nfold != test_id ]
    lst_y = [ y for i, y in enumerate(labels) if i%nfold != test_id ]
    test_x = [ x for i, x in enumerate(data) if i%nfold == test_id ]
    test_y = [ y for i, y in enumerate(labels) if i%nfold == test_id ]
    perm = list(range(len(lst_x)))
    random.shuffle(perm)
    M = int(len(lst_x)*0.9)
    train_x = [ lst_x[i] for i in perm[:M] ]
    train_y = [ lst_y[i] for i in perm[:M] ]
    valid_x = [ lst_x[i] for i in perm[M:] ]
    valid_y = [ lst_y[i] for i in perm[M:] ]
    return train_x, train_y, valid_x, valid_y, test_x, test_y

def cv_split2(data, labels, nfold, valid_id):
    assert (nfold > 1) and (valid_id >= 0) and (valid_id < nfold)
    train_x = [ x for i, x in enumerate(data) if i%nfold != valid_id ]
    train_y = [ y for i, y in enumerate(labels) if i%nfold != valid_id ]
    valid_x = [ x for i, x in enumerate(data) if i%nfold == valid_id ]
    valid_y = [ y for i, y in enumerate(labels) if i%nfold == valid_id ]
    return train_x, train_y, valid_x, valid_y

def pad(sequences, sos=None, eos=None, pad_token='<pad>', pad_left=True, reverse=False):
    ''' input sequences is a list of text sequence [[str]]
        pad each text sequence to the length of the longest
    '''
    if reverse:
        if eos is not None:
            sequences = [[eos] + seq for seq in sequences]
        if sos is not None:
            sequences = [seq + [sos] for seq in sequences]
    else:
        if sos is not None:
            sequences = [[sos] + seq for seq in sequences]
        if eos is not None:
            sequences = [seq + [eos] for seq in sequences]
    max_len = max(5,max(len(seq) for seq in sequences))
    if pad_left:
        return [ [pad_token]*(max_len-len(seq)) + seq for seq in sequences ]
    return [ seq + [pad_token]*(max_len-len(seq)) for seq in sequences ]


def create_one_batch(x, y, map2id, oov='<oov>', gpu=False,
                   sos=None, eos=None, bidirectional=False):
    oov_id = map2id[oov]
    x_padded = pad(x, sos=sos, eos=eos, pad_left=True)
    length = len(x_padded[0])
    batch_size = len(x_padded)
    x_fwd = [ map2id.get(w, oov_id) for seq in x_padded for w in seq ]
    x_fwd = torch.LongTensor(x_fwd)
    assert x_fwd.size(0) == length*batch_size
    x_fwd, y = x_fwd.view(batch_size, length).t().contiguous(), torch.LongTensor(y)

    if gpu:
        x_fwd, y = x_fwd.cuda(), y.cuda()
    if bidirectional:
        for seq in x:
            seq.reverse()
        x_bwd = pad(x, sos=sos, eos=eos, pad_left=True, reverse=True)
        x_bwd = [ map2id.get(w, oov_id) for seq in x_bwd for w in seq ]
        x_bwd = torch.LongTensor(x_bwd)
        x_bwd = x_bwd.view(batch_size, length).t().contiguous()
        if gpu:
            x_bwd = x_bwd.cuda()
        return (x_fwd, x_bwd), y

    return (x_fwd), y, x_padded

# assumes we pad on the left, with zero padding.
def pad_bert(sequences):
    padding = [0.0] * len(sequences[0][0])
    max_len = max(5,max(len(seq) for seq in sequences))
    return [ [padding]*(max_len-len(seq)) + seq for seq in sequences ]


def create_one_batch_bert(x, y, gpu=False):
    x_fwd = pad_bert(x)
    length = len(x_fwd[0])
    batch_size = len(x_fwd)
    x_fwd = [ w for seq in x_fwd for w in seq ]
    x_fwd = torch.Tensor(x_fwd)
    assert x_fwd.size(0) == length*batch_size
    x_fwd, y = x_fwd.view(batch_size, length, x_fwd.size(1)).permute(1,0,2).contiguous(), torch.LongTensor(y)
    if gpu:
        x_fwd, y = x_fwd.cuda(), y.cuda()
    return (x_fwd), y


# shuffle training examples and create mini-batches
def create_batches(x, y, batch_size, map2id, perm=None, sort=False, gpu=False,
                   sos=None, eos=None, bidirectional=False, bert_embed=False, get_text_batches=False):

    lst = perm or list(range(len(x)))
    # sort sequences based on their length; necessary for SST
    if sort:
        lst = sorted(lst, key=lambda i: len(x[i]))

    # reordering based on lst
    x = [ x[i] for i in lst ]
    y = [ y[i] for i in lst ]

    txt_batches = None
    if get_text_batches and not bert_embed:
        txt_batches = []

    sum_len = 0.0
    batches_x = [ ]
    batches_y = [ ]

    size = batch_size
    nbatch = (len(x)-1) // size + 1
    for i in range(nbatch):
        if not bert_embed:
            bx, by, padded_x = create_one_batch(x[i*size:(i+1)*size], y[i*size:(i+1)*size],
                                      map2id, gpu=gpu, sos=sos, eos=eos, bidirectional=bidirectional)
        else:
            padded_x = None
            bx, by = create_one_batch_bert(x[i*size:(i+1)*size], y[i*size:(i+1)*size], gpu=gpu)

        sum_len += len(bx[0])
        batches_x.append(bx)
        batches_y.append(by)

        if get_text_batches and not bert_embed:
            txt_batches.append(padded_x)

    if sort:
        perm = list(range(nbatch))
        random.shuffle(perm)
        batches_x = [ batches_x[i] for i in perm ]
        batches_y = [ batches_y[i] for i in perm ]
        if get_text_batches:
            txt_batches = [txt_batches[i] for i in perm]

    # sys.stdout.write("{} batches, avg len: {:.1f}\n".format(
    #     nbatch, sum_len/nbatch
    # ))
    return batches_x,  batches_y, txt_batches


def load_embedding_npz(path):
    data = np.load(path)
    return [ w.decode('utf8') for w in data['words'] ], data['vals']

def load_embedding_txt(path):
    file_open = gzip.open if path.endswith(".gz") else open
    words = [ ]
    vals = [ ]
    with file_open(path) as fin:
        fin.readline()
        for line in fin:
            line = line.strip()
            if line:
                parts = line.split()
                words.append(parts[0])
                vals += [ float(x) for x in parts[1:] ]
    return words, np.asarray(vals).reshape(len(words),-1)

def load_embedding(path):
    if path.endswith(".npz"):
        return load_embedding_npz(path)
    else:
        return load_embedding_txt(path)

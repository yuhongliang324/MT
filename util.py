__author__ = 'yuhongliang324'

import os
from collections import defaultdict
import nltk
import numpy as np

data_root = '../data'
train_en = os.path.join(data_root, 'train.en-de.low.filt.en')
train_de = os.path.join(data_root, 'train.en-de.low.filt.de')
valid_en = os.path.join(data_root, 'valid.en-de.low.en')
valid_de = os.path.join(data_root, 'valid.en-de.low.de')
test_en = os.path.join(data_root, 'test.en-de.low.en')
test_de = os.path.join(data_root, 'test.en-de.low.de')
blind_de = os.path.join(data_root, 'blind.en-de.low.de')

toy_train_en = os.path.join(data_root, 'toy.train.en')
toy_train_de = os.path.join(data_root, 'toy.train.de')
toy_test_en = os.path.join(data_root, 'toy.test.en')
toy_test_de = os.path.join(data_root, 'toy.test.de')


def read_file(file_name, threshold=1, target=False):
    reader = open(file_name)
    lines = reader.readlines()
    reader.close()
    sentences = map(lambda x: x.strip().split(), lines)
    tok_count = defaultdict(int)

    for sent in sentences:
        for tok in sent:
            tok_count[tok] += 1

    tok_ID = defaultdict(int)
    if not target:  # Unknown words are set to ID = 0 for source language
        ID_tok = {0: '<UNKNOWN>'}
        curID = 1
    else:
        ID_tok = {}
        curID = 0

    for tok, cnt in tok_count.items():
        if cnt < threshold:
            continue
        tok_ID[tok] = curID
        ID_tok[curID] = tok
        curID += 1
    tok_ID['<S>'] = curID
    ID_tok[curID] = '<S>'
    curID += 1
    tok_ID['</S>'] = curID
    ID_tok[curID] = '</S>'
    curID += 1

    num_sent = len(sentences)
    sentVecs = [None for _ in xrange(num_sent)]
    for i in xrange(num_sent):
        sent = ['<S>'] + sentences[i] + ['</S>']
        num_tok = len(sent)
        vec = [0 for _ in xrange(num_tok)]
        for j in xrange(num_tok):
            vec[j] = tok_ID[sent[j]]
        sentVecs[i] = vec

    vocSize = curID
    return tok_ID, ID_tok, sentVecs, vocSize


def read_test_file(file_name, tok_ID=None):
    reader = open(file_name)
    lines = reader.readlines()
    reader.close()
    if tok_ID is None:
        sentences = map(lambda x: x.strip(), lines)
        return sentences
    sentences = map(lambda x: x.strip().split(), lines)

    num_sent = len(sentences)
    sentVecs = [None for _ in xrange(num_sent)]
    for i in xrange(num_sent):
        sent = ['<S>'] + sentences[i] + ['</S>']
        num_tok = len(sent)
        vec = [0 for _ in xrange(num_tok)]
        for j in xrange(num_tok):
            vec[j] = tok_ID[sent[j]]
        sentVecs[i] = vec
    return sentVecs


def sort_by_length(src_vecs, tgt_vecs):
    import random

    def bylen(a, b):
        if a[2] != b[2]:
            return a[2] - b[2]
        return a[3] - b[3]
    lens = [len(vec) for vec in src_vecs]
    num_sent = len(src_vecs)
    ind = range(num_sent)
    random.shuffle(ind)
    cb = zip(src_vecs, tgt_vecs, lens, ind)
    cb.sort(cmp=bylen)
    src_vecs = [item[0] for item in cb]
    tgt_vecs = [item[1] for item in cb]
    return src_vecs, tgt_vecs


def compute_length_prob(src_vecs,tgt_vecs):
    print "Computing length prob for beam search!"
    num_sent = len(src_vecs)
    LP = np.zeros((120,120))
    for i in range(num_sent):
        ls, lt = len(src_vecs[i]), len(tgt_vecs[i])
        LP[ls, lt] += 1
    LP /= float(num_sent)  # add smoothing
    return LP


# For target batch: pad at the end
def make_pad(vecs, padID):
    num_vec = len(vecs)
    lengths = [len(vec) for vec in vecs]
    maxLen = max(lengths)
    pads = [[padID for _ in xrange(num_vec)] for _ in xrange(maxLen)]
    for i in xrange(num_vec):
        for j in xrange(lengths[i]):
            pads[j][i] = vecs[i][j]
    return pads, lengths, maxLen


# For source batch: pad at both sides
def make_pad_bidirection(vecs, startID, stopID):
    num_vec = len(vecs)
    lengths = [len(vec) for vec in vecs]
    maxLen = max(lengths)
    pads = [[stopID for _ in xrange(num_vec)] for _ in xrange(maxLen)]
    for i in xrange(num_vec):
        dif = maxLen - lengths[i]
        num_startID = (dif + 1) // 2
        for j in xrange(num_startID):
            pads[j][i] = startID
        for j in xrange(lengths[i]):
            pads[j + num_startID][i] = vecs[i][j]
    return pads, lengths, maxLen

def eval_BLEU(hypothesis, reference): # hypothesis and reference are list of tokens
    BLEUscore = nltk.translate.bleu_score.bleu([reference],hypothesis,weights=(0.25, 0.25,0.25,0.25))
    return BLEUscore

def test1():
    tok_ID, ID_tok, sentVecs, vocSize = read_file(train_en)
    for i in xrange(10):
        print sentVecs[i]
    print vocSize

# def calculate_length_prior_Prob(src_vecs, tgt_vecs):



if __name__ == '__main__':
    test1()

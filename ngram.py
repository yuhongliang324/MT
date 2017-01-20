__author__ = 'yuhongliang324'
import os
from collections import defaultdict
import math

root = 'en-de/'
train_en = os.path.join(root, 'train.en-de.en')
train_de = os.path.join(root, 'train.en-de.de')
val_en = os.path.join(root, 'valid.en-de.en')
val_de = os.path.join(root, 'valid.en-de.de')


def load_file(fn):
    reader = open(fn)
    lines = reader.readlines()
    reader.close()
    lines = map(lambda x: x.strip(), lines)
    sentences = map(lambda x: x.split(' '), lines)
    return sentences


def calc_bigram(fn):
    sentences = load_file(fn)
    bigram_count = defaultdict(int)  # P(w_t|w_t-1)
    word_count = defaultdict(int)
    for sent in sentences:
        length = len(sent)
        for i in xrange(length - 1):
            bigram = (sent[i], sent[i + 1])
            bigram_count[bigram] += 1
        for i in xrange(length):
            word = sent[i]
            word_count[word] += 1
    total_words = 0
    for cnt in word_count.values():
        total_words += cnt
    word_prob = defaultdict(float)
    for word, count in word_count.items():
        word_prob[word] = float(count) / total_words
    bigram_prob = defaultdict(float)
    for bigram, count in bigram_count.items():
        word_tm1, word_t = bigram
        bigram_prob[bigram] = float(count) / float(word_count[word_tm1])
    for bigram, count in bigram_count.items():
        word_tm1 = bigram[0]
        # print bigram, count, word_count[word_tm1]
    return bigram_prob, word_prob


def calc_PPL(fn, bigram_prob, word_prob, alpha_1, alpha_2):
    alpha_unk = 1. - alpha_1 - alpha_2
    sentences = load_file(fn)
    logP = 0.
    total_length = 0
    # P_unk = 1. / len(word_prob)
    P_unk = 1e-7
    for sent in sentences:
        P_e = (1 - alpha_unk) * word_prob[sent[0]] + alpha_unk * P_unk
        logP += math.log(P_e, 2)
        length = len(sent)
        total_length += length
        for i in xrange(length - 1):
            bigram = (sent[i], sent[i + 1])
            P_cond = alpha_2 * bigram_prob[bigram] + alpha_1 * word_prob[sent[i + 1]] + alpha_unk * P_unk
            logP += math.log(P_cond, 2)
    ppl = math.pow(2., -logP / total_length)
    print ppl


def test1():
    alpha_1, alpha_2 = 0.245, 0.735
    # alpha_1, alpha_2 = 0., 0.98
    bigram_prob, word_prob = calc_bigram(train_en)
    calc_PPL(val_en, bigram_prob, word_prob, alpha_1, alpha_2)


def test2():
    alpha_1, alpha_2 = 0.3395, 0.6305
    bigram_prob, word_prob = calc_bigram(train_de)
    calc_PPL(val_de, bigram_prob, word_prob, alpha_1, alpha_2)


if __name__ == '__main__':
    test1()
    '''
    alphas = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
    alphas = alphas[::-1]
    for alpha in alphas:
        print alpha,
        bigram_prob, word_prob = calc_bigram(train_en)
        calc_PPL(val_en, bigram_prob, word_prob, alpha)'''


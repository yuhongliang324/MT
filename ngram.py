__author__ = 'yuhongliang324'
import os
from collections import defaultdict

root = '../en-de/'
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
    bigram_count = defaultdict(int)
    word_count = defaultdict(int)
    for sent in sentences:
        length = len(sent)
        for i in xrange(length - 1):
            bigram = (sent[i], sent[i + 1])
            bigram_count[bigram] += 1
        for i in xrange(length):
            word = sent[i]
            word_count[word] += 1
    print len(bigram_count), len(word_count)


if __name__ == '__main__':
    calc_bigram(train_en)




__author__ = 'yuhongliang324'
from collections import defaultdict
import sys, os
import math


dn = os.path.dirname(os.path.abspath(__file__))

data_root = os.path.join(dn, '../data')
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


def extract(src_file, tgt_file, alignment_file, out_file, threshold=0, len_threshold=None):
    def extract_sent(src_sent, tgt_sent, fids, eids):
        F = src_sent.split()
        E = tgt_sent.split()
        lenE = len(E)
        BP = set()
        for i1 in xrange(lenE):
            for i2 in xrange(i1, lenE):
                TP = []
                for eid, fid in zip(eids, fids):
                    if i1 <= eid <= i2:
                        TP.append(fid)
                if len(TP) == 0:
                    continue
                TP.sort()
                is_consecutive = True
                j1, j2 = TP[0], TP[-1]
                for fid in xrange(j1, j2 + 1):
                    if fid in TP or fid not in fids:
                        continue
                    is_consecutive = False
                    break
                if not is_consecutive:
                    continue
                SP = []
                for eid, fid in zip(eids, fids):
                    if j1 <= fid <= j2:
                        SP.append(eid)
                if len(SP) == 0:
                    continue
                SP.sort()
                if not (i1 <= SP[0] and SP[-1] <= i2):
                    continue
                subE = ' '.join(E[i1: i2 + 1])
                subF = ' '.join(F[j1: j2 + 1])
                if len(subE) == 0 or len(subF) == 0:
                    continue
                if len_threshold:
                    if len(subE.split()) <= len_threshold and len(subF.split()) <= len_threshold:
                        BP.add(subF + '\t' + subE)
                else:
                    BP.add(subF + '\t' + subE)
        return BP
    reader = open(src_file)
    src_sents = reader.readlines()
    src_sents = map(lambda x: x.strip(), src_sents)
    reader.close()
    reader = open(tgt_file)
    tgt_sents = reader.readlines()
    reader.close()
    tgt_sents = map(lambda x: x.strip(), tgt_sents)

    reader = open(alignment_file)
    alignments = reader.readlines()
    alignments = map(lambda x: x.strip(), alignments)
    reader.close()

    size = len(src_sents)

    count = 0
    bp_count = defaultdict(float)
    for src_sent, tgt_sent, align in zip(src_sents, tgt_sents, alignments):
        sps = align.split()
        fids, eids = [], []
        for sp in sps:
            tmp = sp.split('-')
            eid = int(tmp[0])
            fid = int(tmp[1])
            eids.append(eid)
            fids.append(fid)
        BP = extract_sent(src_sent, tgt_sent, fids, eids)
        for bp in BP:
            bp_count[bp] += 1
        count += 1
        if count % 100 == 0:
            print count, '/', size

    bp_count = sorted(bp_count.iteritems(), key=lambda d: d[1], reverse=True)

    if threshold < 2:
        bp_count_new = bp_count
    else:
        index = 0
        while bp_count[index][1] >= threshold:
            index += 1
        bp_count_new = bp_count[:index]
    E_count = defaultdict(float)
    for item in bp_count_new:
        E = item[0].split('\t')[1]
        count = item[1]
        E_count[E] += count

    bp_prob = defaultdict(float)
    for item in bp_count_new:
        bp = item[0]
        E = item[0].split('\t')[1]
        count = item[1]
        prob = count / E_count[E]
        bp_prob[bp] = prob
    bp_prob = sorted(bp_prob.iteritems(), key=lambda d: d[1], reverse=True)

    writer = open(out_file, 'w')
    for item in bp_prob:
        bp, prob = item[0], item[1]
        nll = -math.log(prob)
        writer.write(bp + '\t' + str(nll) + '\n')
    writer.close()


def test1():
    align_file = 'output/alignment.txt'
    bp_file = 'output/phrase.txt'
    extract(train_de, train_en, align_file, bp_file, threshold=2)


# python $SCRIPT_DIR/phrase-extract.py $TRAIN_DATA.de $TRAIN_DATA.en $OUT_DIR/alignment.txt $OUT_DIR/phrase.txt
extract(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], threshold=2, len_threshold=3)

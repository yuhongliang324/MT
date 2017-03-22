# _*_ coding:utf-8 _*_
import sys
from collections import defaultdict
import math
from train_model1 import IBM1


class IBM2():
    def __init__(self, bitext, output_file, max_iter=10):
        
        ibm1 = IBM1(bitext, output_file='output/alignment1.txt',max_iter=5)
        ibm1.train()
        self.theta = ibm1.theta
        self.align_para = defaultdict(
                lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(
                    lambda: 0.0))))

        for aligned_sent in bitext:
            lt = len(aligned_sent[1])
            ls = len(aligned_sent[0])
            initial_value = 1 / float(lt+1)
            for i in range(0, ls):
                for j in range(0, lt):
                    self.align_para[i][j][lt][ls] = initial_value

        self.min_prob = 1 / 1000000.0
        self.epsion = 1 / 50.0

        self.train(bitext, max_iter)
        self.align(bitext,output_file)


    def train(self, bitext, max_iter):

        for i in range(0, max_iter):
            count_t_s = defaultdict(lambda: defaultdict(float))
            count_given_s = defaultdict(float)

            # count of i given j, l, m
            alignment_count = defaultdict(
                lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(
                    lambda: 0.0))))
            alignment_count_i = defaultdict(
                lambda: defaultdict(lambda: defaultdict(
                    lambda: 0.0)))

            for aligned_sent in bitext:
                trg_sentence = aligned_sent[1]
                src_sentence = aligned_sent[0]  # 1-indexed
                lt = len(aligned_sent[1])
                ls = len(aligned_sent[0])
                total_count = defaultdict(float)

                # E step (a): Compute normalization factors to weigh counts
                for j in range(0, lt):
                    t = trg_sentence[j]
                    total_count[t] = 0
                    for i in range(0, ls):
                        s = src_sentence[i]

                        count = self.theta[s][t] * self.align_para[i][j][lt][ls]
                        total_count[t] += count

                # E step (b): Collect counts
                for j in range(0, lt):
                    t = trg_sentence[j]
                    for i in range(0, ls):
                        s = src_sentence[i]
                        count = self.theta[s][t] * self.align_para[i][j][lt][ls]
                        normalized_count = count / total_count[t]
                        count_t_s[t][s] += normalized_count
                        count_given_s[s] += normalized_count
                        alignment_count[i][j][lt][ls] += normalized_count
                        alignment_count_i[j][lt][ls] += normalized_count

            for s_k, s_dict in self.theta.iteritems():
                for t_k in s_dict:
                    try:
                        estimate = count_t_s[t_k][s_k] / count_given_s[s_k]
                        self.theta[s_k][t_k] = max(estimate, self.min_prob)
                    except:
                        self.theta[s_k][t_k] = self.min_prob
                    # self.theta[e_k][f_k] = self.bi_count[e_k][f_k]/self.e_count[e_k]

            for aligned_sent in bitext:
                lt = len(aligned_sent[1])
                ls = len(aligned_sent[0])
                for j in range(0, ls):
                    for i in range(0, lt):
                        estimate = alignment_count[j][i][lt][ls] / alignment_count_i[i][lt][ls]
                        self.align_para[j][i][lt][ls] = max(estimate,self.min_prob)

            self.calculate_prob(bitext)

    def calculate_prob(self, bitext):

        ll = 0.0
        for idx, (t0, t1) in enumerate(bitext):
            ll += math.log(self.epsion)
            for j in range(len(t1)):
                tmp = 0.0
                for i in range(len(t0)):
                    try:
                        tmp += math.log(self.theta[t0[i]][t1[j]] * self.align_para[j][i][len(t1)][len(t0)])
                    except:
                        pass
                ll += tmp
        ll /= (idx + 1.0)
        # print "[{}] Log Likelihood : {}".format(iter, ll)
        print self.theta["mit"]["with"]

    def align(self, bitext, output_file):

        write_file = open(output_file, "w")
        for idx, (t0, t1) in enumerate(bitext):
            results = []
            lt = len(t1)
            ls = len(t0)
            for i, src_word in enumerate(t0):
                # Initialize trg_word to align with the NULL token
                best_prob = 0
                bestalign = None
                for j, tgt_word in enumerate(t1):
                    align_prob = self.theta[src_word][tgt_word] * self.align_para[j][i][lt][ls]
                    if align_prob >= best_prob:
                        best_prob = align_prob
                        bestalign = j
                results.append("{}-{}".format(bestalign, i))
            line = " ".join(results)
            write_file.write(line + "\n")


def read_bitext(source_file, target_file):
    s_lines = open(source_file,"r").readlines()
    t_lines = open(target_file,"r").readlines()
    s_sentences = map(lambda x: x.strip().split(),s_lines)
    t_sentences = map(lambda x: x.strip().split(), t_lines)
    bitext = zip(s_sentences, t_sentences)
    return bitext

if __name__ == "__main__":
    # source_file = sys.argv[1]
    # target_file = sys.argv[2]
    source_file = "en-de/valid.en-de.low.de"
    target_file = "en-de/valid.en-de.low.en"
    output_file = "output/alignment_ibm2.txt"

    bitext = read_bitext(source_file,target_file)
    model = IBM2(bitext,output_file)

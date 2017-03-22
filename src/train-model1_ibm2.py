'''
IBM1-model
Arguments: (1): source_file (2) target_file (3) output_file
'''
# _*_ coding:utf-8 _*_
import sys
from collections import defaultdict, Counter
# from util import *
import math
import codecs
from train_model1 import IBM1


class IBM2():
    def __init__(self, bitext, output_file, max_iter=10):

        # Get initial translation probability distribution
        # from a few iterations of Model 1 training.
        ibm1 = IBM1(bitext, output_file='output/alignment',max_iter=5)
        ibm1.train()
        self.theta = ibm1.theta
        self.alignment_table = defaultdict(
                lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(
                    lambda: 0.0))))

        # Initialize the distribution of alignment probability,
        # a(i | j,l,m) = 1 / (l+1) for all i, j, l, m
        for aligned_sentence in bitext:
            l = len(aligned_sentence[1])
            m = len(aligned_sentence[0])
            initial_value = 1 / float(l+1)
            for i in range(0, l):
                for j in range(0, m):
                    self.alignment_table[i][j][l][m] = initial_value

        self.min_prob = 1 / 1000000.0
        self.epsion = 1 / 1000000.0

        self.train(bitext, max_iter)
        self.__align(bitext,output_file)


    def train(self, bitext, max_iter):

        for i in range(0, max_iter):
            count_t_given_s = defaultdict(lambda: defaultdict(float))
            count_any_t_given_s = defaultdict(float)

            # count of i given j, l, m
            alignment_count = defaultdict(
                lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(
                    lambda: 0.0))))
            alignment_count_for_any_i = defaultdict(
                lambda: defaultdict(lambda: defaultdict(
                    lambda: 0.0)))

            for aligned_sentence in bitext:
                src_sentence = aligned_sentence[1]
                trg_sentence = aligned_sentence[0]  # 1-indexed
                l = len(aligned_sentence[1])
                m = len(aligned_sentence[0])
                total_count = defaultdict(float)

                # E step (a): Compute normalization factors to weigh counts
                for j in range(0, m):
                    t = trg_sentence[j]
                    total_count[t] = 0
                    for i in range(0, l):
                        s = src_sentence[i]
                        count = self.theta[t][s] * self.alignment_table[i][j][l][m]
                        total_count[t] += count

                # E step (b): Collect counts
                for j in range(0, m):
                    t = trg_sentence[j]
                    for i in range(0, l):
                        s = src_sentence[i]
                        count = self.theta[t][s] * self.alignment_table[i][j][l][m]
                        normalized_count = count / total_count[t]

                        count_t_given_s[t][s] += normalized_count
                        count_any_t_given_s[s] += normalized_count
                        alignment_count[i][j][l][m] += normalized_count
                        alignment_count_for_any_i[j][l][m] += normalized_count

            # M step: Update probabilities with maximum likelihood estimates
            # for s in count_any_t_given_s:
            #     for t in count_t_given_s:
            #         estimate = count_t_given_s[t][s] / count_any_t_given_s[s]
            #         self.theta[s][t] = max(estimate,self.min_prob)

            for e_k, e_dict in self.theta.iteritems():
                for f_k in e_dict:
                    try:
                        estimate = count_t_given_s[e_k][f_k] / count_any_t_given_s[f_k]
                        self.theta[e_k][f_k] = max(estimate, self.min_prob)
                    except:
                        self.theta[e_k][f_k] = self.min_prob
                    # self.theta[e_k][f_k] = self.bi_count[e_k][f_k]/self.e_count[e_k]

            for aligned_sentence in bitext:
                l = len(aligned_sentence[1])
                m = len(aligned_sentence[0])
                for i in range(0, l):
                    for j in range(0, m):
                        estimate = alignment_count[i][j][l][m] / alignment_count_for_any_i[j][l][m]
                        self.alignment_table[i][j][l][m] = max(estimate,self.min_prob)

            # (4) Calculate log data likelihood (Equation 106)
            self.prob_t_a_given_s(bitext)

    def prob_t_a_given_s(self, bitext):
        """
        Probability of target sentence and an alignment given the
        source sentence
        """
        ll = 0.0
        for idx, (t0, t1) in enumerate(bitext):
            ll += math.log(self.epsion)
            # ll -= float(len(t1)) * math.log(float(len()) + 1)
            for j in range(len(t1)):
                tmp = 0.0
                for i in range(len(t0)):
                    try:
                        tmp += math.log(self.theta[t0[i]][t1[j]] * self.alignment_table[j][i][len(t1)][len(t0)])
                    except:
                        pass
                ll += math.log(tmp)
                # print self.theta[e[i]][f[j]], e[i], f[j]
        ll /= (idx + 1.0)
        print "[{}] Log Likelihood : {}".format(iter, ll)
        print self.theta["mit"]["with"]


    def __align_all(self, bitext):
        for sentence_pair in bitext:
            self.__align(sentence_pair)

    def __align(self, bitext, output_file):

        write_file = open(output_file, "w")
        for idx, (t0, t1) in enumerate(bitext):
            results = []
            l = len(t1)
            m = len(t0)
            for j, trg_word in enumerate(t0):
                # Initialize trg_word to align with the NULL token
                best_prob = 0
                best_alignment_point = None
                for i, src_word in enumerate(t1):
                    align_prob = self.theta[trg_word][src_word] * self.alignment_table[i][j][l][m]
                    if align_prob >= best_prob:
                        best_prob = align_prob
                        best_alignment_point = i
                #
                # best_alignment.append((j, best_alignment_point))
                results.append("{}-{}".format(best_alignment_point, j))
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
    #source_file = sys.argv[1]
    #target_file = sys.argv[2]
    source_file = "en-de/valid.en-de.low.de"
    target_file = "en-de/valid.en-de.low.en"
    output_file = "output/alignment_ibm2.txt"

    bitext = read_bitext(source_file,target_file)
    model = IBM2(bitext,output_file)

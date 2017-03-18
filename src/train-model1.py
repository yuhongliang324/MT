'''
IBM1-model
Arguments: (1): source_file (2) target_file (3) output_file
'''
# _*_ coding:utf-8 _*_
import sys
from collections import defaultdict, Counter
from util import *
import math

class IBM1():
    def __init__(self, bitext, output_file, max_iter=10):
        self.bitext = bitext
        self.write_file = output_file
        self.max_iter = max_iter
        self.theta = defaultdict(lambda: defaultdict(float))
        self.bi_count = defaultdict(lambda: defaultdict(float))
        self.e_count = defaultdict(float)
        self.e_size = 1000.0
        self.epsion = 1.0/50

    def train(self):
        # (1) Intitial theta[i][j] = 1/ e_vocab_size (Equation 100)
        for idx, (e, f) in enumerate(self.bitext):
            for i in range(len(e)):
                self.e_count[e[i]] += 1
                for j in range(len(f)):
                    self.theta[e[i]][f[j]] = 1/self.e_size

        # (2) [E] C[i,j] = theta[i,j] / sigma_i theta[i,j] (Equation 110)
        self.bi_count = defaultdict(lambda: defaultdict(float))
        for iter in range(self.max_iter):
            for idx, (e, f) in enumerate(self.bitext):
                for j in range(len(f)):
                    norm = 0.0
                    for i in range(len(e)):
                        norm += self.theta[e[i]][f[j]]
                    for i in range(len(e)):
                        self.bi_count[e[i]][f[j]] += (self.theta[e[i]][f[j]])/norm

            # (3) [M] theta[i,j] =  C[i,j] / sigma_j C[i,j] (Equation 107)
            for e_k, e_dict in self.theta.iteritems():
                for f_k in e_dict:
                    self.theta[e_k][f_k] = self.bi_count[e_k][f_k]/self.e_count[e_k]
                    #if self.theta[e_k][f_k] > 1:
                    #    print e_k, f_k, self.theta[e_k][f_k], self.bi_count[e_k][f_k], self.e_count[e_k]
            # (4) Calculate log data likelihood (Equation 106)
            ll = 0.0
            for idx, (e, f) in enumerate(self.bitext):
                ll += math.log(self.epsion)
                ll -= float(len(f)) * math.log(float(len(e))+1)
                for j in range(len(f)):
                    tmp = 0.0
                    for i in range(len(e)):
                        tmp += self.theta[e[i]][f[j]]
                        #print self.theta[e[i]][f[j]], e[i], f[j]
            ll /= (idx+1.0)
            print "[{}] Log Likelihood : {}".format(iter,round(ll,5))


    def align(self):
        write_file = open(self.write_file,"w")
        for idx, (e, f) in enumerate(self.bitext):
            results = []
            for i in range(len(e)):
                max_j, max_p = -1, 0.0
                for j in range(len(f)):
                    if self.theta[e[i]][f[j]] > max_p:
                        max_j = j
                        max_p = self.theta[e[i]][f[j]]
                results.append("{}-{}".format(max_j,i))
            line = " ".join(results)
            write_file.write(line+"\n")

if __name__ == "__main__":
    #source_file = sys.argv[1]
    #target_file = sys.argv[2]
    source_file = "../data/toy.train.de"
    target_file = "../data/toy.train.en"
    output_file = "./output/alignment.txt"

    bitext = read_bitext(source_file,target_file)
    model = IBM1(bitext,output_file)
    model.train()
    model.align()
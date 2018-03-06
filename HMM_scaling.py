import numpy as np
from numpy import log
from numpy import exp

class HMM:
    data = list()    
    states = list()
    S = 2
    V = 6
    dice = np.zeros([S, 2, V])
    transition = np.zeros([S, S])
    L = 10000
    vit = np.zeros([S, L])
    ptr = np.zeros([L, S])
    vit_expected_z = np.zeros([L])
    F = np.zeros([S, L])
    sc_f = np.zeros([L])
    B = np.zeros([S, L])
    sc_b = np.zeros([L])

    def __init__(self):
        for s in range(self.S):
            for v in range(self.V):
                self.dice[s, 0, v] = v+1
        for v in range(self.V):
            self.dice[0, 1, v] = log(1/self.V)
        for v in range(self.V):
            if(v in [0,1,2,3,4]):
                self.dice[1, 1, v] = log(0.1)
            else:
                self.dice[1, 1, v] = log(0.5)
        self.transition[0,0] = log(0.95)
        self.transition[0,1] = log(0.05)
        self.transition[1,0] = log(0.10)
        self.transition[1,1] = log(0.90)

    def run(self):
        self.Viterbi()
        self.Forward_algorithm()
        self.Backward_algorithm()

    def Viterbi(self):
        for k in range(self.S):
            if(k == 0):
                self.vit[k,0] = 0
            else:
                self.vit[k,0] = -10000000

        for i in range(1, self.L):
            for l in range(self.S):
                max_k = 0
                for s in range(1, self.S):
                    if(self.vit[s,i-1] + self.transition[s,l] > self.vit[max_k,i-1] + self.transition[max_k,l]):
                        max_k = s
                self.ptr[i,l] = max_k
                x = np.where(self.dice[l,0] == self.data[i])
                self.vit[l, i] = self.dice[l,1,x] + self.vit[max_k,i-1] + self.transition[max_k,l]

        max_k = 0
        for l in range(1, self.S):
            if(self.vit[l, self.L-1] > self.vit[max_k, self.L-1]):
                max_k = l
        self.vit_expected_z[self.L-1] = max_k

        for i in range(self.L-2, -1, -1):
            self.vit_expected_z[i] = self.ptr[i+1, self.vit_expected_z[i+1]]
        #self.calc_vit_accuracy()
    
    def calc_vit_accuracy(self):
        correct_list = list()
        for i in range(self.L):
            if(self.states[i] == self.vit_expected_z[i]):
                correct_list.append(1)
            else:
                correct_list.append(0)
        correct = 0
        for i in range(self.L):
            if(correct_list[i] == 1):
                correct += 1
        self.vit_accuracy = correct/self.L*100

    def Forward_algorithm(self):
        for k in range(self.S):
            if(k == 0):
                self.F[k,0] = 1
            else:
                self.F[k,0] = 0
        self.sc_f[0] = 1
        for i in range(1, self.L):
            self.sc_f[i] = 0
            for l in range(self.S):
                x = np.where(self.dice[l,0] == self.data[i])
                sum_fa = 0
                for k in range(self.S):
                    sum_fa += self.F[k, i-1] * exp(self.transition[k,l])
                self.sc_f[i] += exp(self.dice[l, 1, x]) * sum_fa
            for l in range(self.S):
                x = np.where(self.dice[l,0] == self.data[i])
                sum_fa = 0
                for k in range(self.S):
                    sum_fa += self.F[k, i-1] * exp(self.transition[k, l])
                self.F[l,i] = (1/self.sc_f[i]) * exp(self.dice[l, 1, x]) * sum_fa

    def Backward_algorithm(self):
        for k in range(self.S):
            if(k == 0):
                self.B[k, self.L-1] = 1
            else:
                self.B[k, self.L-1] = 0
        self.sc_b[self.L-1] = 1
        for i in range(self.L-2, -1, -1):
            self.sc_b[i] = 0
            for l in range(self.S):
                x = np.where(self.dice[l,0] == self.data[i])
                sum_fa = 0
                for k in range(self.S):
                    sum_fa += self.B[k, i+1] * exp(self.transition[k, l])
                self.sc_b[i] += exp(self.dice[l, 1, x]) * sum_fa
            for l in range(self.S):
                x = np.where(self.dice[l,0] == self.data[i])
                sum_fa = 0
                for k in range(self.S):
                    sum_fa += self.B[k, i+1] * exp(self.transition[k, l])
                self.B[l,i] = (1/self.sc_b[i]) * exp(self.dice[l, 1, x]) * sum_fa

    def load_data(self):
        lines = open('data/sample.txt','r').readlines()
        line = lines[0].split()
        for i in range(len(line)):
            self.data.append(int(line[i]))
        line = lines[1].split()
        for i in range(len(line)):
            self.states.append(int(line[i]))

    def write_data(self):
        #print('accuracy(Viterbi) : ' + str(self.vit_accuracy) + '%')
        output = open('result/viterbi.txt', 'w')
        for i in range(self.L):
            output.write(str(int(self.vit_expected_z[i])) + ' ')

        output = open('result/Forward_scaling.txt', 'w')
        px = 0
        for i in range(self.L):
            px += log(self.sc_f[i])
        output.write('log p(x) = sum(log(s_j)) = ' + str(px) + '\n')
        output.write('p(x) = ' + str(exp(px)))
        
        output = open('result/Backward_scaling.txt', 'w')
        px = 0
        for i in range(self.L):
            px += log(self.sc_b[i])
        output.write('log p(x) = sum(log(s_j)) = ' + str(px) + '\n')
        output.write('p(x) = ' + str(exp(px)))
    

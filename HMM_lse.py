import numpy as np
from numpy import log
from numpy import exp

class HMM:
    iter = 0
    data = list()    
    states = list()
    S = 3
    V = 10
    dice = np.zeros([S, 2, V])
    transition = np.zeros([S, S])
    L = 1461
    vit = np.zeros([S, L])
    ptr = np.zeros([L, S])
    vit_expected_z = np.zeros([L])
    vit_accuracy = 0
    F = np.zeros([S, L])
    For = 0
    B = np.zeros([S, L])
    p_pi = np.zeros([L, S])
    dec_expected_z = np.zeros([L])
    dec_accuracy = 0
    old_LL = 0

    def __init__(self, x):
        self.iter = x

        for s in range(self.S):
            for v in range(self.V):
                self.dice[s, 0, v] = v
                #self.dice[s, 1, v] = np.random.rand()
        self.dice[0,1,0] = 0.01
        self.dice[0,1,1] = 0.01
        self.dice[0,1,2] = 0.01
        self.dice[0,1,3] = 0.02
        self.dice[0,1,4] = 0.10
        self.dice[0,1,5] = 0.15
        self.dice[0,1,6] = 0.20
        self.dice[0,1,7] = 0.20
        self.dice[0,1,8] = 0.15
        self.dice[0,1,9] = 0.15
        self.dice[1,1,0] = 0.02
        self.dice[1,1,1] = 0.03
        self.dice[1,1,2] = 0.10
        self.dice[1,1,3] = 0.15
        self.dice[1,1,4] = 0.20
        self.dice[1,1,5] = 0.20
        self.dice[1,1,6] = 0.15
        self.dice[1,1,7] = 0.10
        self.dice[1,1,8] = 0.03
        self.dice[1,1,9] = 0.02
        self.dice[2,1,0] = 0.15
        self.dice[2,1,1] = 0.20
        self.dice[2,1,2] = 0.20
        self.dice[2,1,3] = 0.20
        self.dice[2,1,4] = 0.15
        self.dice[2,1,5] = 0.03
        self.dice[2,1,6] = 0.02
        self.dice[2,1,7] = 0.02
        self.dice[2,1,8] = 0.02
        self.dice[2,1,9] = 0.01

        for s in range(self.S):
            sum = 0
            for v in range(self.V):
                sum += self.dice[s,1,v]
            for v in range(self.V):
                self.dice[s,1,v] = log(self.dice[s,1,v]/sum)
        
        self.transition[0,0] = 0.6
        self.transition[0,1] = 0.4
        self.transition[0,2] = 0.0
        self.transition[1,0] = 0.2
        self.transition[1,1] = 0.5
        self.transition[1,2] = 0.3
        self.transition[2,0] = 0.0
        self.transition[2,1] = 0.2
        self.transition[2,2] = 0.8

        for s in range(self.S):
            #for ss in range(self.S):
            #    self.transition[s,ss] = np.random.rand()
            sum = 0
            for ss in range(self.S):
                sum += self.transition[s,ss]
            for ss in range(self.S):
                self.transition[s,ss] = log(self.transition[s,ss]/sum)

    def run(self):
        self.Forward_algorithm()
        for i in range(1000):
            print(str(i+1))
            self.Backward_algorithm()
            self.Update_Parameter()
            self.old_LL = self.For
            self.Forward_algorithm()
            if(abs(float(self.For - self.old_LL)) < 0.0000000000001):
                break
            print('Improve : ' + str(float(self.For-self.old_LL)))
            print('dice1')
            print(exp(self.dice[0,1]))
            print('dice2')
            print(exp(self.dice[1,1]))
            print('transition')
            print(exp(self.transition))
            print('\n')
        self.Viterbi()

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
            self.vit_expected_z[int(i)] = self.ptr[int(i)+1, self.vit_expected_z[int(i)+1]]
        self.calc_vit_accuracy()
    
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
                self.F[k,0] = 0
            else:
                self.F[k,0] = -1000000000000
        
        for i in range(1, self.L):
            for l in range(self.S):
                lse = self.logsumexp(self.F, i-1, l)
                x = np.where(self.dice[l,0] == self.data[i])
                self.F[l,i] = self.dice[l,1,x] + lse
        
        max_k = 0
        for k in range(1,self.S):
            if(self.F[k, self.L-1] > self.F[max_k, self.L-1]):
                max_k = k
        sum = 0
        for k in range(self.S):
            sum += exp(self.F[k, self.L-1] - self.F[max_k, self.L-1])
        self.For = self.F[max_k, self.L-1] + log(sum)

    def Backward_algorithm(self):
        for k in range(self.S):
            if(k == 0):
                self.B[k, self.L-1] = 0
            else:
                self.B[k, self.L-1] = -100000000000
        for i in range(self.L-2, -1, -1):
            for k in range(self.S):
                x = np.where(self.dice[k,0] == self.data[i+1])
                max_l = 0
                for l in range(self.S):
                    if(self.transition[k,l] + self.dice[l,1,x] + self.B[l,i+1] > self.transition[k,max_l] + self.dice[max_l,1,x] + self.B[max_l,i+1]):
                        max_l = l
                lse = 0
                for l in range(self.S):
                    lse += exp(self.transition[k,l] + self.dice[l,1,x] + self.B[l,i+1] - self.transition[k,max_l] - self.dice[max_l,1,x] - self.B[max_l,i+1])
                self.B[k,i] = self.transition[k,max_l] + self.dice[max_l,1,x] + self.B[max_l,i+1] + log(lse)

    def Update_Parameter(self):
        Ekb = np.zeros([self.S, self.V])
        for k in range(self.S):
            for b in range(self.V):
                max_i = 0
                for i in range(1,self.L):
                    if(self.data[i] == b+1):
                        if((self.F[k,i] + self.B[k,i] > self.F[k,max_i] + self.B[k,max_i]) or (max_i == 0)):
                            max_i = i
                lse = 0
                for i in range(self.L):
                    if(self.data[i] == b+1):
                        lse += exp(self.F[k,i] + self.B[k,i] - self.F[k, max_i] - self.B[k, max_i])
                Ekb[k,b] = self.F[k,max_i] + self.B[k, max_i] + log(lse) - self.For
        #print(Ekb)

        Akl = np.zeros([self.S, self.S])
        for k in range(self.S):
            for l in range(self.S):
                max_i = 0
                for i in range(1,self.L-1):
                    if(self.F[k,i] + self.dice[l,1,self.data[i+1]-1] + self.B[l,i+1]\
                       > self.F[k,max_i] + self.dice[l,1,self.data[max_i+1]-1] + self.B[l,max_i+1]):
                        max_i = i
                lse = 0
                for i in range(self.L-1):
                    lse += exp(self.F[k,i] + self.dice[l,1,self.data[i+1]-1] + self.B[l,i+1]\
                               - self.F[k,max_i] - self.dice[l,1,self.data[max_i+1]-1] - self.B[l,max_i+1])
                Akl[k,l] = self.F[k,max_i] + self.transition[k,l] + self.dice[l,1,self.data[max_i+1]-1] + self.B[l,max_i+1] + log(lse) - self.For
        #print(Akl)

        for k in range(self.S):
            max_l = 0
            for l in range(1,self.S):
                if(Akl[k,l] > Akl[k,max_l]):
                    max_l = l
            lse = 0
            for l in range(self.S):
                lse += exp(Akl[k,l] - Akl[k,max_l])
            sum = Akl[k, max_l] + log(lse)
            for l in range(self.S):
                self.transition[k,l] = Akl[k,l] - sum
        
        for k in range(self.S):
            max_b = 0
            for b in range(1,self.V):
                if(Ekb[k,b] > Ekb[k,max_b]):
                    max_b = b
            lse = 0
            for b in range(self.V):
                lse += exp(Ekb[k,b] - Ekb[k,max_b])
            sum = Ekb[k, max_b] + log(lse)
            for b in range(self.V):
                self.dice[k,1,b] = Ekb[k,b] - sum

    def logsumexp(self, mat, i, l):
        max_k = 0
        for k in range(1, self.S):
            if(mat[k,i] + self.transition[k,l] > mat[max_k,i] + self.transition[max_k,l]):
                max_k = k
        sum = 0
        for k in range(self.S):
            sum += exp(mat[k,i] + self.transition[k,l] - mat[max_k,i] - self.transition[max_k,l])
        lse = mat[max_k,i] + self.transition[max_k,l] + log(sum)
        return lse

    def Decoding(self):
        for i in range(self.L):
            for k in range(self.S):
                component = self.F[k, i] + self.B[k, i] - self.For
                self.p_pi[i, k] = (exp(component))
        self.dec_expected_z = self.p_pi.argmax(1)
        self.calc_dec_accuracy()

    def calc_dec_accuracy(self):
        correct_list = list()
        for i in range(self.L):
            if(self.states[i] == self.dec_expected_z[i]):
                correct_list.append(1)
            else:
                correct_list.append(0)
        correct = 0
        for i in range(self.L):
            if(correct_list[i] == 1):
                correct += 1
        self.dec_accuracy = correct/self.L*100        

    def load_data(self):
        lines = open('data/sample' + str(self.iter) + '.txt','r').readlines()
        line = lines[0].split()
        for i in range(len(line)):
            self.data.append(int(line[i]))
        line = lines[1].split()
        for i in range(len(line)):
            self.states.append(int(line[i]))

    def write_data(self):
        """
        output = open('result/accuracy/accuracy' + str(self.iter) + '.txt', 'w')
        output.write(str(self.vit_accuracy) + '\n')
        output.write(str(self.dec_accuracy) + '\n')
        """

        output = open('result/viterbi.txt', 'w')
        for i in range(self.L):
            output.write(str(int(self.vit_expected_z[i])) + ' ')
        """
        output = open('result/paramter.txt', 'w')
        output.write('Expected dice parameter...\n\n')
        for s in range(self.S):
            output.write('dice' + str(s) + ':\n')
            for v in range(self.V):
                output.write(str(self.dice[s,0,v]) + '\t')
            output.write('\n')
            for v in range(self.V):
                output.write(str(exp(self.dice[s,1,v])) + '\t')
            output.write('\n\n')
        output.write('transition probability...\n\n')
        for k in range(self.S):
            output.write('state' + str(k) + ':\n')
            for l in range(self.S):
                output.write(str(exp(self.transition[k,l])) + '\t')
            output.write('\n')
        """

        """
        output = open('result/Forward_lse.txt', 'w')
        output.write('log p(x) = log p(x, pi_L = Fair) + log p(x, pi_L = Loaded) = ' + str(float(self.For)) + '\n')
        output.write('p(x) = ' + str(exp(float(self.For))))
        """

        """
        output = open('result/Backward_lse.txt', 'w')
        max_k = 0
        for k in range(1,self.S):
            x = np.where(self.dice[k,0] == self.data[0])
            if(self.dice[k, 1, x] + self.B[k, 0] > self.dice[max_k, 1, x] + self.B[max_k, 0]):
                max_k = k
        sum = 0
        for l in range(self.S):
            x = np.where(self.dice[l,0] == self.data[0])
            sum += exp(self.dice[l, 1, x] + self.B[l, 0] - self.dice[max_k, 1, x] - self.B[max_k, 0])
        x = np.where(self.dice[max_k,0] == self.data[0])
        Back = self.dice[max_k, 1, x] + self.B[max_k, 0] + log(sum)
        output.write('log p(x) = logsumexp( log(a_0l) +　log(el(x0)) + log(bl(0)) ) = ' + str(float(Back)) + '\n')
        output.write('p(x) = ' + str(exp(float(Back))))
        """

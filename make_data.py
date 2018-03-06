import sys
import numpy as np

def main():
    args = sys.argv
    #args[1] : iter num
    iter = int(args[1])
    lines = open('data/format.txt', 'r').readlines()
    index = 0
    for line in lines:
        if(index == 0):
            states = int(line[:-1])
            transition = np.zeros([states,states])
            observed = list()
            probability = list()
            trans = list()
            for i in range(states):
                observed.append(2+i)
            for i in range(states):
                probability.append(observed[-1]+1+i)
            for i in range(states):
                trans.append(probability[-1]+1+i)
        if(index == 1):
            vocab = int(line[:-1])
            dice = np.zeros([states, 2, vocab])
        if(index in observed):
            line = line.split()
            for i in range(vocab):
                dice[index-2, 0, i] = line[i]
        if(index in probability):
            line = line.split()
            for i in range(vocab):
                dice[index-observed[-1]-1, 1, i] = float(line[i])
        if(index in trans):
            line = line.split()
            for i in range(states):
                transition[index-probability[-1]-1, i] = float(line[i])
        index += 1
    column = cast(dice, transition, 1461, states, vocab)
    write_data(column, iter)

def cast(dice, transition, length, states, vocab):
    column = list()
    current_state = 0
    for i in range(length):
        z = np.random.multinomial(1,dice[current_state, 1])
        for i in range(vocab):
            if(z[i] == 1):
                break
        column.append(current_state)
        column.append(int(dice[current_state, 0, i]))
        state = np.random.multinomial(1,transition[current_state])
        for i in range(states):
            if(state[i] == 1):
                break
        current_state = i

    return column

def write_data(column, iter):
    output = open('data/sample' + str(iter) +'.txt', 'w')
    for i in range(len(column)):
        if(i % 2 == 1):
            output.write(str(column[i]) + ' ')
    output.write('\n')
    for i in range(len(column)):
        if(i % 2 == 0):
            output.write(str(column[i]) + ' ')
    output.write('\n')

if(__name__ == '__main__'):
    main()

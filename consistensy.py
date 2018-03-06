
def main():
    correct_path = list()
    viterbi_path = list()
    correct = open('data/correct_path.txt', 'r').read()
    correct_path = correct.split(' ')
    viterbi = open('result/viterbi.txt', 'r').read()
    viterbi_path = viterbi.split(' ')
    consist = 0
    for i in range(len(correct_path)):
        if(correct_path[i] == viterbi_path[i]):
            consist += 1
    print(str(consist/len(correct_path)*100))

if (__name__ == '__main__'):
    main()

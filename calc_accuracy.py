
def main():
    lines = open('data/sample1_300.txt', 'r').readlines()
    c300 = lines[1]
    lines = open('data/sample1_30000.txt', 'r').readlines()
    c30000 = lines[1]
    lines = open('result/viterbi_300.txt', 'r').readlines()
    v300 = lines[0]
    lines = open('result/viterbi_30000.txt', 'r').readlines()
    v30000 = lines[0]
    check300 = 0
    check30000 = 0
    for i in range(300):
        if(c300[i] == v300[i]):
            check300 += 1
    for i in range(30000):
        if(c30000[i] == v30000[i]):
            check30000 += 1
    check300 /= 3
    check30000 /= 300
    output = open('result/accuracy.txt','w')
    output.write('accuracy...\n')
    output.write('300   : ' + str(check300) + '%\n')
    output.write('30000 : ' + str(check30000) + '%\n')

if(__name__ == '__main__'):
    main()

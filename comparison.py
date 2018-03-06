import numpy as np

def main():
    accuracy = load_data()
    mean_accuracy = list()
    mean_accuracy.append(accuracy[0].mean()) 
    mean_accuracy.append(accuracy[1].mean())
    times = comparison(accuracy)
    write_data(mean_accuracy, times)

def load_data():
    accuracy = np.zeros([2,100])
    for i in range(1, 101):
        lines = open('result/accuracy/accuracy' + str(i) + '.txt', 'r').readlines()
        line = lines[0][:-1]
        accuracy[0, i-1] = float(line)
        line = lines[1][:-1]
        accuracy[1, i-1] = float(line)
    return accuracy

def comparison(accuracy):
    times = 0
    for i in range(100):
        if(accuracy[0, i] > accuracy[1, i]):
            times += 1
    return times

def write_data(mean_accuracy, times):
    output = open('result/mean_accuracy.txt', 'w')
    output.write('Viterbi  : ' + str(mean_accuracy[0]) + '%\n')
    output.write('Decoding : ' + str(mean_accuracy[1]) + '%\n')
    output.write('Viterbi wins  : ' + str(times) + 'times\n')
    output.write('Decoding wins : ' + str(100-times) + 'times\n')

if __name__ == '__main__':
    main()

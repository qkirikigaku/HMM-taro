import numpy as np
import matplotlib.pyplot as plt

def main():
    data = open('./result/viterbi.txt', 'r').read()
    condition  = data[:-1].split(' ')
    data = open('./data/sample1.txt', 'r').readline()
    score = data[:-2].split(' ')
    print(condition)
    print(score)
    for i in range(len(condition)):
        condition[i] = 2 - int(condition[i])
    for i in range(len(score)):
        score[i] = int(score[i])

    left = np.arange(365)
    bar_height = np.array(score)
    line_height = np.array(condition)


    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.bar(left, bar_height[0:365], align = 'center', color = 'royalblue', linewidth = 0)
    ax1.set_ylabel('Score')
    ax1.set_xlim([0,365])
    ax1.set_ylim([0,9])
    ax2 = fig.add_subplot(212)
    ax2.bar(left, line_height[0:365], align = 'center', linewidth = 0, color = 'crimson')
    ax2.set_ylabel('Condition')
    ax2.set_xlim([0,365])
    ax2.set_ylim([0.0,2.0])
    fig.savefig('./1st.png', dpi=300)
    plt.close(1)

    fig = plt.figure()

    left = np.arange(365,730)
    ax1 = fig.add_subplot(211)
    ax1.bar(left, bar_height[365:730], align = 'center', color = 'royalblue', linewidth = 0)
    ax1.set_ylabel('Score')
    ax1.set_xlim([365,730])
    ax1.set_ylim([0,9])
    ax2 = fig.add_subplot(212)
    ax2.bar(left, line_height[365:730], linewidth = 0, color = 'crimson', align = 'center')
    ax2.set_ylabel('Condition')
    ax2.set_xlim([365,730])
    ax2.set_ylim([0.0,2.0])
    fig.savefig('./2nd.png', dpi=300)
    plt.close(1)

    fig = plt.figure()

    left = np.arange(730,1095)
    ax1 = fig.add_subplot(211)
    ax1.bar(left, bar_height[730:1095], align = 'center', color = 'royalblue', linewidth = 0)
    ax1.set_ylabel('Score')
    ax1.set_xlim([730,1095])
    ax1.set_ylim([0,9])
    ax2 = fig.add_subplot(212)
    ax2.bar(left, line_height[730:1095], linewidth = 0, color = 'crimson', align = 'center')
    ax2.set_ylabel('Condition')
    ax2.set_xlim([730,1095])
    ax2.set_ylim([0.0,2.0])
    fig.savefig('./3rd.png', dpi=300)
    plt.close(1)

    fig = plt.figure()

    left = np.arange(1095,1461)

    ax1 = fig.add_subplot(211)
    ax1.bar(left, bar_height[1095:1461], align = 'center', color = 'royalblue', linewidth = 0)
    ax1.set_ylabel('Score')
    ax1.set_xlim([1095,1461])
    ax1.set_ylim([0,9])
    ax2 = fig.add_subplot(212)
    ax2.bar(left, line_height[1095:1461], linewidth = 0, color = 'crimson', align = 'center')
    ax2.set_ylabel('Condition')
    ax2.set_xlim([1095,1461])
    ax2.set_ylim([0.0,2.0])
    fig.savefig('./4th.png', dpi=300)

if(__name__ == '__main__'):
    main()

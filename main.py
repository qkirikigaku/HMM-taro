import sys

def main():
    args = sys.argv
    #args[1] : logsumexp or scaling [2] : iteration num
    if(args[1] == 'logsumexp'):
        from HMM_lse import HMM
    elif(args[1] == 'scaling'):
        from HMM_scaling import HMM
    else:
        print('Definition Error : args[1] == logsumexp or scaling')
        return 1
    iter = int(args[2])
    temp_object = HMM(iter)
    temp_object.load_data()
    temp_object.Viterbi()
    temp_object.write_data()
    
if(__name__ == '__main__'):
    main()

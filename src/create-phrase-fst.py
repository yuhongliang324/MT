from collections import defaultdict
import codecs
import sys

def main(argv):
    fout = codecs.open(argv[2], 'w','utf-8')
    state_connect = defaultdict(dict)  # {0:{('un','<eps>'):4,}}
    statenum = 0

    NULL = u'<eps>'
    fin = codecs.open(argv[1], 'r', 'utf-8')
    for phrase in fin:
        itemli = phrase.strip().split(u'\t')
        source = itemli[0].split(u' ')
        target = itemli[1].split(u' ')
        try:
            prob = itemli[2]
        except:
            print phrase
        laststate = 0
        for item in source:
            if (item, NULL) in state_connect[laststate]:
                laststate = state_connect[laststate][(item, NULL)]
            else:
                statenum += 1
                state_connect[laststate][(item, NULL)] = statenum
                fout.write(
                    str(laststate) + u' ' + str(state_connect[laststate][(item, NULL)]) + u' ' + item + u' ' + NULL + u'\n')
                laststate = statenum
        for item in target:
            if (NULL, item) in state_connect[laststate]:
                laststate = state_connect[laststate][(NULL, item)]
            else:
                statenum += 1
                state_connect[laststate][(NULL, item)] = statenum
                fout.write(
                    str(laststate) + u' ' + str(state_connect[laststate][(NULL, item)]) + u' ' + NULL + u' ' + item + u'\n')
                laststate = statenum
        fout.write(
            str(laststate) + u' ' + str(0) + u' ' + NULL + u' ' + NULL + u' ' + prob + u'\n')

    """
    Add special token and ending state.
    """
    fout.write(
        str(0) + u' ' + str(0) + u' ' + u'</s>' + u' ' + u'</s>' + u'\n')
    fout.write(
        str(0) + u' ' + str(0) + u' ' + u'<unk>' + u' ' + u'<unk>' + u'\n')
    fout.write(
        str(0) + u'\n')

    fout.close()
    fin.close()

if __name__ == "__main__":
    # argv = ['','phrase_example.txt','phrase-fst.txt']
    main(sys.argv)


from collections import defaultdict

fout = open('phrase-fst-tem.txt', 'w')
state_connect = defaultdict(dict)  # {0:{('un','<eps>'):4,}}
statenum = 0

NULL = '<eps>'
for phrase in open('phrase.txt'):
    itemli = phrase.strip().split('\t')
    source = itemli[0].split(' ')
    target = itemli[1].split(' ')
    prob = itemli[2]
    laststate = 0
    for item in source:
        if (item, NULL) in state_connect[laststate]:
            laststate = state_connect[laststate][(item, NULL)]
        else:
            statenum += 1
            state_connect[laststate][(item, NULL)] = statenum
            fout.write(
                str(laststate) + ' ' + str(state_connect[laststate][(item, NULL)]) + ' ' + item + ' ' + NULL + '\n')
            laststate = statenum
    for item in target:
        if (NULL, item) in state_connect[laststate]:
            laststate = state_connect[laststate][(NULL, item)]
        else:
            statenum += 1
            state_connect[laststate][(NULL, item)] = statenum
            fout.write(
                str(laststate) + ' ' + str(state_connect[laststate][(NULL, item)]) + ' ' + NULL + ' ' + item + '\n')
            laststate = statenum
    fout.write(
        str(laststate) + ' ' + str(0) + ' ' + NULL + ' ' + NULL + ' ' + prob + '\n')

"""
Add special token and ending state.
"""
fout.write(
    str(0) + ' ' + str(0) + ' ' + '</s>' + ' ' + '</s>' + '\n')
fout.write(
    str(0) + ' ' + str(0) + ' ' + '<unk>' + ' ' + '<unk>' + '\n')
fout.write(
    str(0) + '\n')

fout.close()

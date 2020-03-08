from nltk.corpus import brown
from nltk import FreqDist, WittenBellProbDist, bigrams
import sys

TRAIN = 10000
TEST = 500


sents = brown.tagged_sents(tagset='universal')
train_sents= sents[:TRAIN]
test_sents = sents[TRAIN:TRAIN + TEST]


words = []
tags = []
total_bigrams = []
for sentence in train_sents:
    sentence = [('S', 'S')] + sentence + [('E', 'E')]
    stags = [t for (_, t) in sentence]
    swords = [w for (w, _) in sentence]
    tags += stags
    words += sentence
    total_bigrams += list(bigrams(stags))
    
tag_bi = FreqDist(total_bigrams)
#wt_count = FreqDist(words)
#wc = FreqDist(words)
#types = list(wc.keys())

tag_count = FreqDist(tags)
ts = list(tag_count.keys())
tn = len(tags)
print(tag_count['X'])
#word_start = [s[0][1] for s in train_sents]
#st_count = FreqDist(word_start)
#init_prob = {}
#init_prob['s'] = WittenBellProbDist(FreqDist(word_start), bins=1e5)


def create_table(x, y):
    tab = []
    for i in range(x):
        tab.append([])
        for j in range(y):
            tab[i].append(0)
    return tab

def build_trans(tagset):
    global tag_bi, tag_count
    smoothed = {}
    for tag in tagset:
        transitions = [t for (o, t) in total_bigrams if o == tag]
        smoothed[tag] = WittenBellProbDist(FreqDist(transitions), bins=1e5)
    
    return smoothed

def build_emis(tagset):
    global wt_count, tag_count, types, words
    smoothed = {}
    for tag in tagset:
        ws = [w for (w, t) in words if t == tag]
        smoothed[tag] = WittenBellProbDist(FreqDist(ws), bins=1e5)
    

    #ems = create_table(len(tagset), len(types))
    #for i in range(len(tagset)):
    #    for j in range(len(types)):

            #ems[i][j] = (wt_count[(types[j], tagset[i])]/tag_count[tagset[i]])
           # ems[i][j] = smoothed[tagset[i]].prob(types[j])
    return smoothed


emis = build_emis(ts)
trans = build_trans(ts)

def viterbi(wseq, tagset=ts, ep = emis, tp = trans):
    global init_probs  
    vitable = create_table(len(tagset), len(wseq))
    backpoints = create_table(len(tagset), len(wseq))


    for t in range(len(tagset)):
        vitable[t][0] = tp['S'].prob(tagset[t]) * ep[tagset[t]].prob(wseq[0])
    #pretty_print(vitable)
    for i in range(1, len(wseq)):
        for q in range(len(tagset)):
            candidates = [(vitable[q_prime][i-1] * tp[tagset[q_prime]].prob(tagset[q]) * ep[tagset[q]].prob(wseq[i]), q_prime) for q_prime in range(len(tagset))]
            vitable[q][i] = max(candidates)[0]
            backpoints[q][i] = max(candidates)[1]
    
    lastpoint = max([(vitable[q][len(wseq) - 1], q) for q in range(len(tagset))])[1] 
    bestPOS = len(wseq) * [0]
    bestPOS[-1] = tagset[lastpoint]
    for i in range(len(wseq) - 1, 0, -1):
        pt = backpoints[lastpoint][i]
        bestPOS[i - 1] = tagset[pt]
        lastpoint=pt

    bestPOS[0] = 'S'
    res = [tag for tag in bestPOS]
#    print(res)
    return res

def eval(tst_s, tagset=ts):
    correct = 0
    count = 0
    confusion_matrix = create_table(len(ts), len(ts))
    for s in tst_s:
        s = [('S', 'S')] + s + [('E', 'E')]
        ws = [w for (w, _) in s]
        tgs = [t for (_, t) in s]
        ptags = viterbi(ws)
        #print("AA : " + str(tgs))
        #print("PP : " + str(ptags))
        for i in range(len(tgs)):
            count += 1
            if tgs[i] == ptags[i]:
                correct +=1
            confusion_matrix[ts.index(tgs[i])][ts.index(ptags[i])] += 1
        
    print("a\p\t" + "\t".join(ts))
    for i in range(len(confusion_matrix)):
        line = ts[i] + "\t"
        for j in range(len(confusion_matrix)):
                line += str(confusion_matrix[i][j])+ "\t"
        print(line)
    print("prediction rate is :" + str((correct * 100)/count))



eval(test_sents)





#viterbi(sys.argv[1].split(" "), ts)

    


        

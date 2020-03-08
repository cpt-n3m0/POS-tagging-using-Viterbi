from nltk.corpus import brown
from nltk import FreqDist, WittenBellProbDist, bigrams
import sys

TRAIN = 10000
TEST = 500

def flatten(sents):
    return [w for s in sents for w in s]



sents = brown.tagged_sents(tagset='universal')
train_sents= sents[:TRAIN]
test_sents = sents[TRAIN:TRAIN + TEST]

flat_sents = flatten(train_sents)
words = [w for (w, _) in flat_sents]
tags = [t for (_, t) in flat_sents]



wc = FreqDist(words)
types = list(wc.keys())

tag_count = FreqDist(tags)
ts = list(tag_count.keys())
tn = len(tags)
print(ts)
word_start = [s[0][1] for s in train_sents]
st_count = FreqDist(word_start)
init_prob = {}
init_prob['s'] = WittenBellProbDist(FreqDist(word_start), bins=1e5)


total_bigrams = []
for sentence in train_sents:
    stags = [t for (_, t) in sentence]
    total_bigrams += list(bigrams(stags))
tag_bi = FreqDist(total_bigrams)
wt_count = FreqDist(flat_sents)



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
        transitions = [t for (o, t) in tag_bi if o == tag]
        smoothed[tag] = WittenBellProbDist(FreqDist(transitions), bins=1e5)
    
    return smoothed

def build_emis(tagset):
    global wt_count, tag_count, types, flat_sents
    smoothed = {}
    for tag in tagset:
        words = [w for (w, t) in flat_sents if t == tag]
        smoothed[tag] = WittenBellProbDist(FreqDist(words), bins=1e5)
    

    #ems = create_table(len(tagset), len(types))
    #for i in range(len(tagset)):
    #    for j in range(len(types)):

            #ems[i][j] = (wt_count[(types[j], tagset[i])]/tag_count[tagset[i]])
           # ems[i][j] = smoothed[tagset[i]].prob(types[j])
    return smoothed



def viterbi(wseq, tagset=ts):
    global init_probs  
    vitable = create_table(len(tagset), len(wseq))
    backpoints = create_table(len(tagset), len(wseq))

    transitions = build_trans(tagset)
    emissions = build_emis(tagset)

    for t in range(len(tagset)):
        vitable[t][0] = init_prob['s'].prob(tagset[t]) * emissions[tagset[t]].prob(wseq[0])
    #pretty_print(vitable)
    for i in range(1, len(wseq)):
        for q in range(len(tagset)):
            candidates = [(vitable[q_prime][i-1] * transitions[tagset[q_prime]].prob(tagset[q]) * emissions[tagset[q]].prob(wseq[i]), q_prime) for q_prime in range(len(tagset))]
            vitable[q][i] = max(candidates)[0]
            backpoints[q][i] = max(candidates)[1]
    
    lastpoint = max([(vitable[q][len(wseq) - 1], q) for q in range(len(tagset))])[1] 
    bestPOS = len(wseq) * [0]
    bestPOS[-1] = tagset[lastpoint]
    for i in range(len(wseq) - 1, 0, -1):
        pt = backpoints[lastpoint][i]
        bestPOS[i - 1] = tagset[pt]
        lastpoint=pt

    res = [tag for tag in bestPOS]
#    print(res)
    return res

def eval(tst_s):
    correct = 0
    count = 0
    for s in tst_s:
        ws = [w for (w, _) in s]

        tgs = [t for (_, t) in s]
        ptags = viterbi(ws)
        print("SENTENCE :" + str(ws))
        print("ORIGINAL :" + str(tgs))
        print("PREDICTED :" + str(ptags))
        for i in range(len(tgs)):
            count += 1
            if tgs[i] == ptags[i]:
                correct +=1
            #else:
             #   print(tags[i] + " was falsely predicted as : " + ptags[i])
    print("prediction rate is :" + str((correct * 100)/count))



eval(test_sents)





#viterbi(sys.argv[1].split(" "), ts)

    


        

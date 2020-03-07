from nltk.corpus import brown
from nltk import FreqDist, WittenBellProbDist, bigrams
import sys

TRAIN = 10000
TEST = 500

def flatten(sents):
    return [w for s in sents for w in s]

def get_tag_bigram_freq():
    tags = bigram([ t for (_, t) in flatten(brown.tagged_sents(tagset='universal'))])


sents = brown.tagged_sents(tagset='universal')
train_sents= sents[:TRAIN]
test_sents = sents[TRAIN:TEST]

flat_sents = flatten(train_sents)
words = [w for (w, _) in flat_sents]
tags = set([t for (_, t) in flat_sents])

init_prob = []



wc = FreqDist(words)
types = list(wc.keys())

tag_count = FreqDist(tags)
ts = list(tag_count.keys())
tn = sum(list(tag_count.values()))

tag_bi = FreqDist(list(bigrams(tags)))
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
   # transitions = len(tagset) * [len(tagset) * [0]]
    transitions = create_table(len(tagset), len(tagset))
    for i in range(len(tagset)):
        for j in range(len(tagset)):
            transitions[i][j] = tag_bi[(tagset[i], tagset[j])]/tag_count[tagset[i]]
    
    return transitions

def build_emis(tagset):
    global wt_count, tag_count, types, tags, flat_sents
    smoothed = {}
    for tag in tags:
        words = [w for (w, t) in flat_sents if t == tag]
        smoothed[tag] = WittenBellProbDist(FreqDist(words), bins=1e5)
    

    ems = create_table(len(tagset), len(types))
    for i in range(len(tagset)):
        for j in range(len(types)):

            #ems[i][j] = (wt_count[(types[j], tagset[i])]/tag_count[tagset[i]])
            ems[i][j] = smoothed[tagset[i]].prob(types[j])
    return ems




def words_to_Is(seq):
    global types
    return [types.index(w) for w in seq]


def pretty_print(l):
    for r in l :
        line = ""
        for c in r:
            line += str(c) + " "*3
        print(line)

def viterbi(wseq, tagset):
    
    #vitable = len(tagset) * [len(wseq) * [0]]
    vitable = create_table(len(tagset), len(wseq))
    bookkeep = len(wseq) * [0]

    transitions = build_trans(tagset)
    emissions = build_emis(tagset)

    # calculate inital prob
    wseq_i  = words_to_Is(wseq)
    print(wseq_i)
    for t in range(len(tagset)):
        vitable[t][0] = (tag_count[tagset[t]]/tn) * emissions[t][wseq_i[0]]

    #pretty_print(vitable)
    for i in range(1, len(wseq)):
        for q in range(len(tagset)):
            candidates = [(vitable[q_prime][i-1] * transitions[q_prime][q] * emissions[q][wseq_i[i]], q_prime) for q_prime in range(len(tagset))]
            vitable[q][i] = max(candidates)[0]
            #bookkeep[i] = max(zip(candidates, range(len(candidates))))[1]
            bookkeep[i] = q
    
    print(bookkeep)
    for k in bookkeep:
        print(tagset[k])
    pretty_print(vitable)
    
viterbi(["I", "ate", "an", "animal"], ts)

    


        

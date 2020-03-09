from nltk.corpus import brown
from nltk import FreqDist, WittenBellProbDist, bigrams
import sys
import re
TRAIN = 10000
TEST = 500


sents = brown.tagged_sents(tagset='universal')
train_sents= sents[:TRAIN]
test_sents = sents[TRAIN:TRAIN + TEST]


words = [w for s in sents for (w, _) in s]

wc = FreqDist(words)
tags = []
total_bigrams = []
wt_pairs = []

nwords= []
for sentence in train_sents:
    sentence = [('S', 'S')] + sentence + [('E', 'E')]
    stags = []
    for i in range(len(sentence)):
        word = sentence[i][0]
        # preprocess
        if wc[word] < 2 and word.endswith('able'):
            sentence[i]= ('UNK-able', sentence[i][1])
        stags += [sentence[i][1]]
        nwords += [sentence[i][0]]
        wt_pairs += [sentence[i]]

    
    total_bigrams += list(bigrams(stags))
    tags += stags
words = nwords    
tag_bi = FreqDist(total_bigrams)
#wt_count = FreqDist(wt_pairs)
#print(wt_pairs)
#types = list(wc.keys())
#def preprocess():
#    global words, wt_pairs
#    for i in range(len(wt_pairs)):
#        w = wt_pairs[i][0]
#        if wc[w] < 3 and w[-3:] == 'ing':
#            words = ['UNK-ing' if wd == w else wd for wd in words]
#            wt_pairs[i] = ('UNK-ing', wt_pairs[i][1])
#    nouns = []
#    for i in range(len(words)):
#        if len(words[i]) > 1 and words[i][0].isupper() and words[i - 1] != 'S':
#            nouns.append(words[i])
#            words[i] = 'N-UNK'
    #wt_pairs = [('N-UNK', t) if wd in nouns else (wd,t) for (wd,t) in wt_pairs]

            
        
#preprocess() 

wc = FreqDist(words)
print(wc['UNK-ing'])
tag_count = FreqDist(tags)
ts = list(tag_count.keys())
tn = len(tags)
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
    global wt_count, tag_count, types, wt_pairs
    smoothed = {}
    for tag in tagset:
        ws = [w for (w, t) in wt_pairs if t == tag]
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

    res = [tag for tag in bestPOS]
#    print(res)
    return res

def eval(tst_s, tagset=ts):
    correct = 0
    count = 0
    confusion_matrix = create_table(len(ts), len(ts))
   # nouns = []
    for s in tst_s:
       # for i in range(len(s)):
       #     if len(s[i][0]) > 1 and s[i][0][0].isupper() and i != 0:
       #         nouns.append(s[i][0])
       #         s[i] = ('N-UNK', s[i][1])
        s = [('UNK-able',t) if w not in words and w.endswith('ing') else (w,t) for (w,t) in s]
        
      #  s = [('NUNK',t) if w not in words and w[-3:] =='ing' else (w,t) for (w,t) in s]
        s = s + [('E', 'E')]
        ws = [w for (w, _) in s]
        tgs = [t for (_, t) in s]
        
       # print([(w, t) for (w, t) in s if w not in list(wc.keys())])
        ptags = viterbi(ws)
       # print(ws)
       # print("OO : " + str(tgs)) 
       # print("PP : " + str(ptags)) 
        for i in range(1, len(tgs)):
            count += 1
            if tgs[i] == ptags[i]:
                correct +=1
            elif ws[i] == 'UNK-ing':
                print(tgs[i] + " and not " + ptags[i])

            confusion_matrix[ts.index(tgs[i])][ts.index(ptags[i])] += 1

        
    #print(nouns)
    #print(len(nouns))
    # print stats    
    print("a\p\t" + "\t".join(ts))
    for i in range(len(confusion_matrix)):
        line = ts[i] + "\t"
        for j in range(len(confusion_matrix)):
                line += str(confusion_matrix[i][j])+ "\t"
        print(line)
    print("prediction rate is :" + str((correct * 100)/count))



eval(test_sents)





#viterbi(sys.argv[1].split(" "), ts)

    


        

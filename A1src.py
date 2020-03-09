from nltk.corpus import brown
from nltk import FreqDist, WittenBellProbDist, bigrams
import sys
import re
TRAIN = 10000
TEST = 500
RARETHRESH = 1

sents = brown.tagged_sents(tagset='universal')
train_sents= sents[:TRAIN]
test_sents = sents[TRAIN:TRAIN + TEST]

# initial counts for preprocessing use
words = [w for s in train_sents for (w, _) in s]
wc = FreqDist(words)


tags = []
total_bigrams = []
wt_pairs = []
nwords= [] #processed words

for sentence in train_sents:
    sentence = [('S', 'S')] + sentence + [('E', 'E')]
    stags = []
    for i in range(len(sentence)):
        word = sentence[i][0]
        #preprocess
        suffixes = ['ing', 'able', 'ible', 'ed', 'ly', 'dom', 'ism', 'ness', 'ist', 'ship', 'al', 'ance', 'tion', 'sion', 'ise']
        if wc[word] <= RARETHRESH :
            if len(word) > 1 and word[0].isupper() and i > 1:
             #   print(" {} {} {}".format(sentence[i-1][0], sentence[i][0], sentence[i + 1][0]))
             #   print(i)
                sentence[i]= ('N-UNK', sentence[i][1])
            for suf in suffixes:
                if word.endswith(suf):
                    sentence[i]= ('UNK-' + suf, sentence[i][1])
 

 
 
        stags += [sentence[i][1]]
        nwords += [sentence[i][0]]
        wt_pairs += [sentence[i]]

    
    total_bigrams += list(bigrams(stags))
    tags += stags
words = nwords    
tag_bi = FreqDist(total_bigrams)
            

wc = FreqDist(words)
print(wc['UNK-ing'])
tag_count = FreqDist(tags)
ts = list(tag_count.keys())
tn = len(tags)



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
    return bestPOS

def eval(tst_s, tagset=ts):
    correct = 0
    count = 0
    confusion_matrix = create_table(len(ts), len(ts))
    for s in tst_s:
        s =[('S', 'S')]+ s + [('E', 'E')]
        
        for i in range(len(s)):
            w = s[i][0]
            if len(w) > 1 and w not in words and w[0].isupper() and i > 1:
                s[i] = ('N-UNK', s[i][1])
            if w not in words:
                for suf in suffixes:
                    if w.endswith(suf):
                        s[i] = ("UNK-" + suf, s[i][1])
        
        ws = [w for (w, _) in s]
        tgs = [t for (_, t) in s]
        
        ptags = viterbi(ws)
        for i in range( len(tgs)):
            count += 1
            if tgs[i] == ptags[i]:
                correct +=1

            confusion_matrix[ts.index(tgs[i])][ts.index(ptags[i])] += 1

    #print stats
    print("a\p\t" + "\t".join(ts))
    for i in range(len(confusion_matrix)):
        line = ts[i] + "\t"
        for j in range(len(confusion_matrix)):
                line += str(confusion_matrix[i][j])+ "\t"
        print(line)
    print("prediction rate is :" + str((correct * 100)/count))



eval(test_sents)





#viterbi(sys.argv[1].split(" "), ts)

    


        

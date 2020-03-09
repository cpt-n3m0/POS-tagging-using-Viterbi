from nltk.corpus import brown
from nltk import FreqDist, WittenBellProbDist, bigrams
import sys
import re
import time


TRAIN = 10000
TEST = 500
RARETHRESH = 1
PREPROCESS = True
PP_CAP = True
PP_SUF = True
suffixes = ['ing', 'able', 'ible', 'ed', 'ly', 'dom', 'ism', 'ness', 'ist', 'ship', 'al', 'ance', 'tion', 'sion', 'ise']

VERBOSE = False





def create_table(x, y):
    tab = []
    for i in range(x):
        tab.append([])
        for j in range(y):
            tab[i].append(0)
    return tab

def build_trans(tagset, tbigrams):
    #global tag_bi, tag_count
    smoothed = {}
    for tag in tagset:
        transitions = [t for (o, t) in tbigrams if o == tag]
        smoothed[tag] = WittenBellProbDist(FreqDist(transitions), bins=1e5)
    
    return smoothed

def build_emis(tagset, wrd_tag_pairs):
    smoothed = {}
    for tag in tagset:
        ws = [w for (w, t) in wrd_tag_pairs if t == tag]
        smoothed[tag] = WittenBellProbDist(FreqDist(ws), bins=1e5)
    
    return smoothed


def viterbi(wseq, tagset, ep , tp ):
    vitable = create_table(len(tagset), len(wseq))
    backpoints = create_table(len(tagset), len(wseq))


    for t in range(len(tagset)):
        vitable[t][0] = tp['S'].prob(tagset[t]) * ep[tagset[t]].prob(wseq[0])

    for i in range(1, len(wseq)):
        for q in range(len(tagset)):
            candidates = [(vitable[q_prime][i-1] * tp[tagset[q_prime]].prob(tagset[q]) * ep[tagset[q]].prob(wseq[i]), q_prime) for q_prime in range(len(tagset))]
            vitable[q][i] = max(candidates)[0]
            backpoints[q][i] = max(candidates)[1]
    
    # trace back steps
    lastpoint = max([(vitable[q][len(wseq) - 1], q) for q in range(len(tagset))])[1] 
    bestPOS = len(wseq) * [0]
    bestPOS[-1] = tagset[lastpoint]
    for i in range(len(wseq) - 1, 0, -1):
        pt = backpoints[lastpoint][i]
        bestPOS[i - 1] = tagset[pt]
        lastpoint=pt
    bestPOS[0] = 'S'

    return bestPOS


def train(train_sents):
    if PREPROCESS:
        # initial counts for preprocessing use
        words = [w for s in train_sents for (w, _) in s]
        wc = FreqDist(words)
        type_count = len(wc.keys())
    train_duration = time.time()
    pp_duration = 0
 
    tags = []
    total_bigrams = []
    wt_pairs = []
    nwords= [] #processed words

    counter = 0
    for sentence in train_sents:
        sentence = [('S', 'S')] + sentence + [('E', 'E')]
        stags = []
        print("Training progress: %d%% \r" % (counter * 100/len(train_sents) + 1), end='', flush=True)
        counter+= 1
        for i in range(len(sentence)):
            if PREPROCESS:
                start = time.time()
                word = sentence[i][0]
                #preprocess
                if wc[word] <= RARETHRESH :
                    if len(word) > 1 and word[0].isupper() and i > 1:
                        sentence[i]= ('N-UNK', sentence[i][1])
                    for suf in suffixes:
                        if word.endswith(suf):
                            sentence[i]= ('UNK-' + suf, sentence[i][1])
                pp_duration += time.time() - start
            stags += [sentence[i][1]]
            nwords += [sentence[i][0]]
            wt_pairs += [sentence[i]]


        total_bigrams += list(bigrams(stags))
        tags += stags
    words = nwords    
    wc = FreqDist(words)
    print("")
    tag_bi_count = FreqDist(total_bigrams)            
    tag_count = FreqDist(tags)
    ts = list(tag_count.keys())

    train_duration = time.time() - train_duration
    if VERBOSE:
        # tag stats
        print("* Training lasted:\t" + str(train_duration) + "ms")
        print("* Preprocessing lasted:\t{}ms ({}%)".format(str(pp_duration), str(int(pp_duration * 100/train_duration))))
        print("* Tags occurences :")
        for t in tag_count:
            if t in ['S', 'E']:
                continue
            print("{}\t{}".format(t, str(tag_count[t])))
        print("* Type count : " + str(type_count))
        
    return wt_pairs, words, wc, tags, total_bigrams, tag_bi_count, ts

def eval(tst_s, tagset, ep , tp):
    if VERBOSE: print("\nStaring Testing ...")
    correct = 0
    count = 0
    confusion_matrix = create_table(len(ts), len(ts))
    
    counter = 0
    unks = {}
    for s in tst_s:
        
        print("Testing progress: %d%% \r" % (counter * 100/len(tst_s) + 1), end='', flush=True)
        counter += 1
        s =[('S', 'S')]+ s + [('E', 'E')]
        
        for i in range(len(s)):
            w = s[i][0]
            if len(w) > 1 and w not in words and w[0].isupper() and i > 1:
                s[i] = ('N-UNK', s[i][1])
            if w not in words:
                for suf in suffixes:
                    if w.endswith(suf):
                        s[i] = ("UNK-" + suf, s[i][1])
                        try:
                            unks[suf] += 1
                        except:
                            unks[suf] = 0

        
        ws = [w for (w, _) in s]
        tgs = [t for (_, t) in s]
        
        ptags = viterbi(ws, tagset, ep, tp)
        for i in range(len(tgs)):
            count += 1
            if tgs[i] == ptags[i]:
                correct +=1

            confusion_matrix[ts.index(tgs[i])][ts.index(ptags[i])] += 1
    print("")
    #print stats
    
    if VERBOSE:
        ntwords= [w for s in train_sents for (w, _) in s]
        ntwc = FreqDist(ntwords)
        print("* Test type count:\t" + str(len(ntwc.keys())))
        print("* UNK occurences (% of total words)")
        for unk in unks:
            print("{}\t{} ({}%)".format(unk, str(unks[unk]), str(unks[unk] * 100/ntwords)))
        print("")

    print("-" * 48 + "Confusion Matrix" + '-' * 48)
    print("a\p\t" + "\t".join(ts))
    for i in range(len(confusion_matrix)):
        line = ts[i] + "\t"
        for j in range(len(confusion_matrix)):
                line += str(confusion_matrix[i][j])+ "\t"
        print(line)
    print("Accuracy rate is :" + str((correct * 100)/count))




def setFlags(cmd):
    global PREPROCESS, PP_CAP, PP_SUF, VERBOSE
    if "-pp" in cmd:
        PREPROCESS = False
    if '-cpp' in cmd:
        PP_CAP = False
    if '-sufpp' in cmd:
        PP_SUF = False
    if '-v' in cmd:
        VERBOSE = True

if __name__ == '__main__':
    setFlags(sys.argv[1:])
    sents = brown.tagged_sents(tagset='universal')
    train_sents= sents[:TRAIN]
    test_sents = sents[TRAIN:TRAIN + TEST]
    
    wt_pairs, words, wc, tags, tbigrams, tag_bi_count, ts = train(train_sents)
    
    eps = build_emis(ts, wt_pairs)
    tps = build_trans(ts, tbigrams)
    
    eval(test_sents, ts, eps, tps)


    


        

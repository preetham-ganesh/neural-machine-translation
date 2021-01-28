import math
import operator
from functools import reduce
from nltk.translate.meteor_score import meteor_score
import numpy as np
import unicodedata
import re
import pandas as pd

def count_ngram(candidate, references, n):
    clipped_count = 0
    count = 0
    r = 0
    c = 0
    for si in range(len(candidate)):
        ref_counts = []
        ref_lengths = []
        for reference in references:
            ref_sentence = reference[si]
            ngram_d = {}
            words = ref_sentence.strip().split()
            ref_lengths.append(len(words))
            limits = len(words) - n + 1
            for i in range(limits):
                ngram = ' '.join(words[i:i+n]).lower()
                if ngram in ngram_d.keys():
                    ngram_d[ngram] += 1
                else:
                    ngram_d[ngram] = 1
            ref_counts.append(ngram_d)
        cand_sentence = candidate[si]
        cand_dict = {}
        words = cand_sentence.strip().split()
        limits = len(words) - n + 1
        for i in range(0, limits):
            ngram = ' '.join(words[i:i + n]).lower()
            if ngram in cand_dict:
                cand_dict[ngram] += 1
            else:
                cand_dict[ngram] = 1
        clipped_count += clip_count(cand_dict, ref_counts)
        count += limits
        r += best_length_match(ref_lengths, len(words))
        c += len(words)
    if clipped_count == 0:
        pr = 0
    else:
        pr = float(clipped_count) / count
    bp = brevity_penalty(c, r)
    return pr, bp

def clip_count(cand_d, ref_ds):
    count = 0
    for m in cand_d.keys():
        m_w = cand_d[m]
        m_max = 0
        for ref in ref_ds:
            if m in ref:
                m_max = max(m_max, ref[m])
        m_w = min(m_w, m_max)
        count += m_w
    return count

def best_length_match(ref_l, cand_l):
    least_diff = abs(cand_l-ref_l[0])
    best = ref_l[0]
    for ref in ref_l:
        if abs(cand_l-ref) < least_diff:
            least_diff = abs(cand_l-ref)
            best = ref
    return best

def brevity_penalty(c, r):
    if c > r:
        bp = 1
    else:
        bp = math.exp(1-(float(r)/c))
    return bp

def geometric_mean(precisions):
    return (reduce(operator.mul, precisions)) ** (1.0 / len(precisions))

def BLEU(candidate, references):
    precisions = []
    for i in range(4):
        pr, bp = count_ngram(candidate, references, i+1)
        precisions.append(pr)
    bleu = geometric_mean(precisions) * bp
    return bleu, geometric_mean(precisions), bp

def text_retrieve(name):
    with open('/home/preetham/Documents/neural-machine-translation/models/results/en-es/'+name, 'r', encoding='utf-8') as f:
        text = f.read()
    f.close()
    return text.split('\n')

def meteor(candidate, reference):
    score = []
    for i, j in zip(candidate, reference):
        s = meteor_score([j], i, 4)
        score.append(s)
    return np.mean(score)

def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')

def data_preprocessing(inp, tar, pred, range_0, range_1):
    new_tar, new_pred = [], []
    for i, j, k in zip(tar, pred, inp):
        if len(k.split(' ')) >= range_0 and len(k.split(' ')) <= range_1:
            i = i.replace('<', '')
            i = i.replace('#', '')
            i = i.replace('>', '')
            j = j.replace('<', '')
            j = j.replace('#', '')
            j = j.replace('>', '')
            new_tar.append(i)
            new_pred.append(j)
            if i == '' or j == '':
                continue
    return new_tar, new_pred

def main():
    print()
    model_name = 'model_1'
    set = 'val'
    inp = text_retrieve(model_name+'/'+set+'-predictions/inp_text.txt')
    tar = text_retrieve(model_name+'/'+set+'-predictions/tar_text.txt')
    pred = text_retrieve(model_name+'/'+set+'-predictions/pred_text.txt')
    ranges = [[0, 9], [10, 19], [20, 29], [30, 39], [40, 100], [0, 100]]
    d = {'range_0': [], 'range_1': [], 'bleu': [], 'precision': [], 'brevity': [], 'meteor': []}
    for i in ranges:
        new_tar, new_pred = data_preprocessing(inp, tar, pred, i[0], i[1])
        bleu, ps, bp = BLEU(new_pred, [new_tar])
        meteor_ = meteor(new_pred, new_tar)
        d['range_0'].append(i[0])
        d['range_1'].append(i[1])
        d['bleu'].append(round(bleu*100, 3))
        d['precision'].append(round(ps*100, 3))
        d['brevity'].append(round(bp*100, 3))
        d['meteor'].append(round(meteor_*100, 3))
    df = pd.DataFrame(d, columns=['range_0', 'range_1', 'bleu', 'precision', 'brevity', 'meteor'])
    print(df)
    print()
    #df.to_csv('/home/preetham/Documents/neural-machine-translation/models/results/en-es/'+model_name+'/history/'+set+'.csv', index=False)

main()

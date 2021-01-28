import re
from collections import Counter
import time

def text_retrieve(name):
    with open('/home/preetham/Documents/neural-machine-translation/models/data/en-es/cleaned/'+name, 'r',
              encoding='utf-8') as f:
        text = f.read()
    f.close()
    return text

def text_stat(text):
    letters = Counter(text)
    print('No. of unique characters in text: ', len(letters.keys()))
    print(letters.keys())
    words = Counter(text.split(' '))
    print('No. of unique words in text: ', len(words.keys()))
    return letters, words

def find_rare_words(words):
    rare_words = []
    for w in words.keys():
        if words[w] == 1 and not w.isdigit() and (len(w) == 8) and w.isalpha():
            rare_words.append(w)
    print('Number of rare words: ', len(rare_words))
    return rare_words

def convert_rare_words(inp_lines, tar_lines, rare_words):
    new_inp_lines, new_tar_lines = [], []
    for i, j in zip(inp_lines, tar_lines):
        t1 = time.time()
        inp_line = i.split(' ')
        tar_line = j.split(' ')
        common_words = list(set(inp_line) & set(rare_words) & set(tar_line))
        if len(common_words) == 0:
            new_inp_lines.append(i)
            new_tar_lines.append(j)
        else:
            for k in common_words:
                i = i.replace(k, '<'+'#'.join(k)+'>')
                j = j.replace(k, '<'+'#'.join(k)+'>')
            new_inp_lines.append(i)
            new_tar_lines.append(j)
    return new_inp_lines, new_tar_lines

def lines_to_text(lines, sep):
    text = ''
    for i in range(len(lines)):
        if i == len(lines) - 1:
            text += str(lines[i])
        else:
            text += str(lines[i]) + sep
    return text

def dataset_save(lines, name):
    text = lines_to_text(lines, '\n')
    f = open('/home/preetham/Documents/neural-machine-translation/models/data/en-es/oov-handled/'+name, 'w',
             encoding='utf-8')
    f.write(text)
    f.close()
    del text

def oov_handling(name, common_rare_words):
    inp_lines = text_retrieve(name+'.en').split('\n')
    tar_lines = text_retrieve(name+'.es').split('\n')
    inp_lines, tar_lines = convert_rare_words(inp_lines, tar_lines, common_rare_words)
    print(name+' set processed')
    dataset_save(inp_lines, name+'.en')
    del inp_lines
    dataset_save(tar_lines, name+'.es')
    del tar_lines
    print(name+' set saved')
    print()
    
def main():
    print()
    text = text_retrieve('train.en')
    text = re.sub('\n', ' ', text)
    letters, words = text_stat(text)
    print()
    en_rare_words = find_rare_words(words)
    print()
    del text, letters, words
    text = text_retrieve('train.es')
    text = re.sub('\n', ' ', text)
    letters, words = text_stat(text)
    print()
    es_rare_words = find_rare_words(words)
    print()
    common_rare_words = list(set(es_rare_words) & set(en_rare_words))
    print('No. of common rare words in both corpuses: ', len(common_rare_words))
    print()
    del text, letters, words, es_rare_words, en_rare_words
    oov_handling('train', common_rare_words)
    oov_handling('val', common_rare_words)
    oov_handling('test', common_rare_words)

main()

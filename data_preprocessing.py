import pandas as pd
import unicodedata
import re
import pickle
from random import shuffle

def text_retrieve(name):
    with open('/home/preetham/Documents/neural-machine-translation/models/data/en-es/original/'+name, 'r',
              encoding='utf-8') as f:
        text = f.read()
    f.close()
    return text.split('\n')

def remove_html_markup(s):
    tag = False
    quote = False
    out = ""
    for c in s:
        if c == '<' and not quote:
            tag = True
        elif c == '>' and not quote:
            tag = False
        elif (c == '"' or c == "'") and tag:
            quote = not quote
        elif not tag:
            out = out + c
    return out

def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')

def preprocess_sentence(w, lang):
    w = remove_html_markup(w)
    w = w.lower().strip()
    if w == '':
        return 0
    else:
        w = w.replace('##at##-##at##', '-')
        w = w.replace('&apos;', "'")
        w = w.replace('&quot;', '"')
        w = w.replace('&#91;', "")
        w = w.replace('&#93;', "")
        w = w.replace('&#124;', "")
        if lang == 'en':
            w = unicode_to_ascii(w)
            w = re.sub(r"[^-!$&(),./%0-9:;?a-z€'\"]+", " ", w)
        else:
            w = re.sub(r"[^-!$&(),./%0-9:;?a-záéíñóúü¿¡€'\"]+", " ", w)
        w = re.sub('\.{2,}', '.', w)
        w = re.sub(r'(\d)th', r'\1 th', w, flags=re.I)
        w = re.sub(r'(\d)st', r'\1 st', w, flags=re.I)
        w = re.sub(r'(\d)rd', r'\1 rd', w, flags=re.I)
        w = re.sub(r'(\d)nd', r'\1 nd', w, flags=re.I)
        punc = list("-!$&(),./%:;?¿¡€'")
        for i in punc:
            w = w.replace(i, " "+i+" ")
        w = w.replace('"', ' " ')
        w = w.strip()
        w = re.sub(r'\s+', ' ', w)
        return w

def create_dataset(inp_lines, tar_lines, n_examples):
    new_inp_lines, new_tar_lines = [], []
    for i, j in zip(inp_lines[:n_examples], tar_lines[:n_examples]):
        if pd.isnull(i) or pd.isnull(j):
            continue
        else:
            inp = preprocess_sentence(i, 'es')
            tar = preprocess_sentence(j, 'en')
            if inp == 0 or tar == 0:
                continue
            else:
                new_inp_lines.append(inp)
                new_tar_lines.append(tar)
    return new_inp_lines, new_tar_lines

def lines_to_text(lines, sep):
    text = ''
    for i in range(len(lines)):
        if i == len(lines) - 1:
            text += str(lines[i])
        else:
            text += str(lines[i]) + sep
    return text

def drop_duplicates(inp_lines, tar_lines):
    d = {'inp': inp_lines, 'tar': tar_lines}
    df = pd.DataFrame(d)
    df = df.drop_duplicates()
    if len(df.inp.unique()) < len(df.tar.unique()):
        df = df.drop_duplicates(subset='inp', keep='first')
    else:
        df = df.drop_duplicates(subset='tar', keep='first')
    return list(df['inp']), list(df['tar'])

def drop_line_length(inp, tar, inp_max_length, tar_max_length):
    new_inp, new_tar = [], []
    for i, j in zip(inp, tar):
        if len(i.split(' ')) <= inp_max_length and len(j.split(' ')) <= tar_max_length:
            new_inp.append(i)
            new_tar.append(j)
    return new_inp, new_tar

def dataset_save(lines, name):
    text = lines_to_text(lines, '\n')
    f = open('/home/preetham/Documents/neural-machine-translation/models/data/en-es/cleaned/'+name, 'w', encoding='utf-8')
    f.write(text)
    f.close()
    
def dataset_preprocessing(inp_lines, tar_lines):
    print('No. of Input lines before preprocessing: ', len(inp_lines))
    print('No. of Target lines before preprocessing: ', len(tar_lines))
    print()
    inp_lines, tar_lines = create_dataset(inp_lines, tar_lines, None)
    print('No. of Input lines after preprocessing sentences: ', len(inp_lines))
    print('No. of Target lines after preprocessing sentences: ', len(tar_lines))
    print()
    inp_lines, tar_lines = drop_duplicates(inp_lines, tar_lines)
    print('No. of Input sentences after dropping duplicates: ', len(inp_lines))
    print('No. of Target sentences after dropping duplicates: ', len(tar_lines))
    print()
    inp_lines, tar_lines = drop_line_length(inp_lines, tar_lines, 40, 40)
    print('No. of Input sentences after dropping sentences greater than 40 tokens: ', len(inp_lines))
    print('No. of Target sentences after dropping sentences greater than 40 tokens: ', len(tar_lines))
    print()
    return inp_lines, tar_lines

def main():
    #dataset_0
    print()
    tr_es_lines, tr_en_lines = [], []
    data_0 = pd.read_csv('/home/preetham/Documents/neural-machine-translation/models/data/en-es/original/dataset_0/spa.txt',
                         sep='\t', encoding='utf-8', names=['en', 'es', 'x'])
    es_lines = list(data_0['es'])
    en_lines = list(data_0['en'])
    print('Dataset 0 details')
    print()
    es_lines, en_lines = dataset_preprocessing(es_lines, en_lines)
    tr_es_lines += es_lines
    tr_en_lines += en_lines
    print('Dataset 0 preprocessed successfully')
    print()
    print('Total no. of Spanish lines in dataset: ', len(tr_es_lines))
    print('Total no. of English lines in dataset: ', len(tr_en_lines))
    print()
    #dataset_1
    es_lines = text_retrieve('dataset_1/europarl-v7.es-en.es')
    en_lines = text_retrieve('dataset_1/europarl-v7.es-en.en')
    print('Dataset 1 details')
    print()
    es_lines, en_lines = dataset_preprocessing(es_lines, en_lines)
    tr_es_lines += es_lines
    tr_en_lines += en_lines
    print('Dataset 2 preprocessed successfully')
    print()
    tr_es_lines, tr_en_lines = drop_duplicates(tr_es_lines, tr_en_lines)
    print('Total no. of Spanish lines in dataset: ', len(tr_es_lines))
    print('Total no. of English lines in dataset: ', len(tr_en_lines))
    print()
    #dataset_2
    for i in range(0, 4):
        es_lines += text_retrieve('dataset_2/train_' + str(i) + '.es')
        en_lines += text_retrieve('dataset_2/train_' + str(i) + '.en')
    print('Dataset 2 details')
    print()
    es_lines, en_lines = dataset_preprocessing(es_lines, en_lines)
    tr_es_lines += es_lines
    tr_en_lines += en_lines
    print('Dataset 2 preprocessed successfully')
    print()
    tr_es_lines, tr_en_lines = drop_duplicates(tr_es_lines, tr_en_lines)
    print('Total no. of Spanish lines in train dataset: ', len(tr_es_lines))
    print('Total no. of English lines in train dataset: ', len(tr_en_lines))
    print()
    #dataset_3
    val_es_lines = text_retrieve('dataset_3/newstest2012.es')
    val_en_lines = text_retrieve('dataset_3/newstest2012.en')
    val_es_lines, val_en_lines = dataset_preprocessing(val_es_lines, val_en_lines)
    print('Total no. of Spanish lines in validation dataset: ', len(val_es_lines))
    print('Total no. of English lines in validation dataset: ', len(val_en_lines))
    print()
    test_es_lines = text_retrieve('dataset_3/newstest2013.es')
    test_en_lines = text_retrieve('dataset_3/newstest2013.en')
    test_es_lines, test_en_lines = dataset_preprocessing(test_es_lines, test_en_lines)
    print('Total no. of Spanish lines in test dataset: ', len(test_es_lines))
    print('Total no. of English lines in test dataset: ', len(test_en_lines))
    print()
    c = list(zip(tr_es_lines, tr_en_lines))
    shuffle(c)
    tr_es_lines, tr_en_lines = zip(*c)
    c = list(zip(val_es_lines, val_en_lines))
    shuffle(c)
    val_es_lines, val_en_lines = zip(*c)
    c = list(zip(test_es_lines, test_en_lines))
    shuffle(c)
    test_es_lines, test_en_lines = zip(*c)
    dataset_save(tr_es_lines, 'train.es')
    dataset_save(tr_en_lines, 'train.en')
    dataset_save(val_es_lines, 'val.es')
    dataset_save(val_en_lines, 'val.en')
    dataset_save(test_es_lines, 'test.es')
    dataset_save(test_en_lines, 'test.en')

main()

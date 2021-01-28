import os
import logging
import tensorflow as tf
from utils import english_to_spanish, text_retrieve

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

def text_save(text, name):
    f = open('/home/preetham/Documents/neural-machine-translation/models/en-es/results/model_2/val-predictions/'+name, 'w', encoding='utf-8')
    f.write(text)
    f.close()

def main():
    test_inp = text_retrieve('data/oov-handled/val.en')
    test_tar = text_retrieve('data/oov-handled/val.es')
    inp_text = ''
    tar_text = ''
    pred_text = ''
    for i in range(len(test_inp)):
        print(i)
        inp = str(test_inp[i])
        tar = str(test_tar[i])
        print('Input sentence: ', inp)
        print('Target sentence: ', tar)
        pred = english_to_spanish(inp)
        print('Model output sentence: ', pred)
        print()
        inp_text += inp + '\n'
        tar_text += tar + '\n'
        pred_text += pred + '\n'
    text_save(inp_text, 'inp_text.txt')
    text_save(tar_text, 'tar_text.txt')
    text_save(pred_text, 'pred_text.txt')

main()

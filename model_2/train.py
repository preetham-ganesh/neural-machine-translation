import tensorflow as tf
import os
import logging
from utils import open_file, save_file, model_training, model_testing, tokenizer
import tensorflow_datasets as tfds

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

def main():
    print()
    train_inp = open_file('data/swt-tokenized/train.en')
    val_inp = open_file('data/swt-tokenized/val.en')
    test_inp = open_file('data/swt-tokenized/test.en')
    train_tar = open_file('data/swt-tokenized/train.es')
    val_tar = open_file('data/swt-tokenized/val.es')
    test_tar = open_file('data/swt-tokenized/test.es')
    print('No. of original sentences in Training set: ', len(train_inp))
    print('No. of original sentences in Validation set: ', len(val_inp))
    print('No. of original sentences in Test set: ', len(test_inp))
    print()
    train_inp, val_inp, test_inp = tokenizer(train_inp, val_inp, test_inp)
    train_tar, val_tar, test_tar = tokenizer(train_tar, val_tar, test_tar)
    batch_size = 128
    loc_from = '/home/preetham/Documents/neural-machine-translation/models/en-es/results/model_2/tokenizer/'
    inp_lang = tfds.deprecated.text.SubwordTextEncoder.load_from_file(loc_from + 'inp-tokenizer')
    tar_lang = tfds.deprecated.text.SubwordTextEncoder.load_from_file(loc_from + 'tar-tokenizer')
    print('Input Vocabulary size: ', inp_lang.vocab_size + 2)
    print('Target Vocabulary size: ', tar_lang.vocab_size + 2)
    print()
    parameters = {'inp_vocab_size': inp_lang.vocab_size + 2, 'tar_vocab_size': tar_lang.vocab_size + 2,
                  'n_layers': 6, 'd_model': 512, 'dff': 2048, 'batch_size': batch_size, 'epochs': 3, 'n_heads': 8,
                  'train_steps_per_epoch': len(train_inp) // batch_size, 'dropout': 0.1,
                  'val_steps_per_epoch': len(val_inp) // batch_size, 'test_steps': len(test_inp) // batch_size,
                  'inp_max_length': 40, 'tar_max_length': 40, 'max_train_steps': 200000}
    save_file(parameters, 'results/model_2/utils/parameters')
    print('No. of Training steps per epoch: ', parameters['train_steps_per_epoch'])
    print('No. of Validation steps per epoch: ', parameters['val_steps_per_epoch'])
    print('No. of Testing steps: ', parameters['test_steps'])
    print()
    train_dataset = tf.data.Dataset.from_tensor_slices((train_inp, train_tar))
    train_dataset = train_dataset.shuffle(len(train_inp)).padded_batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((val_inp, val_tar))
    val_dataset = val_dataset.shuffle(len(val_inp)).padded_batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_inp, test_tar))
    test_dataset = test_dataset.shuffle(len(test_inp)).padded_batch(batch_size)
    model_training(train_dataset, val_dataset, parameters)
    model_testing(test_dataset, parameters)

main()

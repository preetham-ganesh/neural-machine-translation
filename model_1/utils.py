import tensorflow as tf
import pickle
import time
import matplotlib.pyplot as plt
from model import Encoder, Decoder
import numpy as np
import re
import sentencepiece as spm
import pandas as pd

def create_new_dataset(inp, tar, inp_max_length, tar_max_length):
    new_inp, new_tar = [], []
    for i, j in zip(inp, tar):
        if len(i.split(' ')) <= inp_max_length and len(j.split(' ')) <= tar_max_length:
            new_inp.append(i)
            new_tar.append(j)
    return new_inp, new_tar
    
def text_retrieve(name):
    with open('/home/preetham/Documents/neural-machine-translation/models/data/en-es/'+name, 'r',
              encoding='utf-8') as f:
        text = f.read()
    f.close()
    return text.split('\n')

def tokenize(train, val, test):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    lang_tokenizer.fit_on_texts(train)
    train_tensor = lang_tokenizer.texts_to_sequences(train)
    train_tensor = tf.keras.preprocessing.sequence.pad_sequences(train_tensor, padding='post')
    val_tensor = lang_tokenizer.texts_to_sequences(val)
    val_tensor = tf.keras.preprocessing.sequence.pad_sequences(val_tensor, padding='post')
    test_tensor = lang_tokenizer.texts_to_sequences(test)
    test_tensor = tf.keras.preprocessing.sequence.pad_sequences(test_tensor, padding='post')
    return lang_tokenizer, train_tensor, val_tensor, test_tensor

def open_file(name):
    loc_to = '/home/preetham/Documents/neural-machine-translation/models/results/en-es/model_1/utils/'
    with open(loc_to + name + '.pkl', 'rb') as f:
        d = pickle.load(f)
    f.close()
    return d

def save_file(d, name):
    loc_to = '/home/preetham/Documents/neural-machine-translation/models/results/en-es/model_1/utils/'
    with open(loc_to + name + '.pkl', 'wb') as f:
        pickle.dump(d, f, protocol=2)
    print(name + ' saved successfully')
    f.close()

def loss_function(real, pred):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)

@tf.function
def train_step(inp, tar, encoder, decoder, optimizer, tar_word_index, batch_size, hidden):
    loss = 0
    with tf.GradientTape() as tape:
        enc_out, enc_hidden = encoder(inp, True, hidden)
        dec_hidden = enc_hidden
        dec_inp = tf.expand_dims([tar_word_index['<s>']] * batch_size, 1)
        for i in range(1, tar.shape[1]):
            prediction, dec_hidden = decoder(dec_inp, dec_hidden, enc_out, True)
            loss += loss_function(tar[:, i], prediction)
            dec_inp = tf.expand_dims(tar[:, i], 1)
    batch_loss = loss / tar.shape[1]
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    train_loss(batch_loss)

def validation_step(inp, tar, encoder, decoder, tar_word_index, batch_size, hidden):
    loss = 0
    enc_out, enc_hidden = encoder(inp, False, hidden)
    dec_hidden = enc_hidden
    dec_inp = tf.expand_dims([tar_word_index['<s>']] * batch_size, 1)
    for i in range(1, tar.shape[1]):
        prediction, dec_hidden = decoder(dec_inp, dec_hidden, enc_out, False)
        loss += loss_function(tar[:, i], prediction)
        dec_inp = tf.expand_dims(tar[:, i], 1)
    batch_loss = loss / tar.shape[1]
    val_loss(batch_loss)

def model_training(train_dataset, val_dataset):
    global train_loss, val_loss
    loc_to = '/home/preetham/Documents/neural-machine-translation/models/results/en-es/model_1/'
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    val_loss = tf.keras.metrics.Mean(name='val_loss')
    parameters = open_file('parameters')
    encoder = Encoder(parameters['emb_size'], parameters['inp_vocab_size'], parameters['rnn_size'], parameters['rate'])
    decoder = Decoder(parameters['emb_size'], parameters['tar_vocab_size'], parameters['rnn_size'], parameters['rate'])
    optimizer = tf.keras.optimizers.Adam()
    checkpoint_dir = loc_to+'training_checkpoints'
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)
    manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=3)
    tar_word_index = open_file('tar-word-index')
    try:
        full_df = pd.read_csv(loc_to+'history/full_steps.csv')
    except:
        full_df = pd.DataFrame(columns=['steps', 'train_loss'])
    try:
        split_df = pd.read_csv(loc_to+'history/split_steps.csv')
    except:
        split_df = pd.DataFrame(columns=['steps', 'train_loss', 'val_loss'])
    step = 0
    step_break = False
    best_val_loss = None
    checkpoint_count = 0
    for epoch in range(parameters['epochs']):
        hidden = encoder.initialize_hidden_state(parameters['batch_size'], parameters['rnn_size'])
        train_loss.reset_states()
        val_loss.reset_states()
        for (batch, (inp, tar)) in enumerate(train_dataset.take(parameters['train_steps_per_epoch'])):
            batch_start = time.time()
            train_step(inp, tar, encoder, decoder, optimizer, tar_word_index, parameters['batch_size'], hidden)
            d = {'steps': int(step), 'train_loss': train_loss.result().numpy()}
            full_df = full_df.append(d, ignore_index=True)
            batch_end = time.time()
            if step % 100 == 0:
                print('Training step=' + str(step) + ', Batch=' + str(batch) + ', Training Loss=' +
                      str(round(train_loss.result().numpy(), 3)) + ', Time taken=' +
                      str(round(batch_end - batch_start, 3)) + ' sec')
                full_df.to_csv(loc_to+'history/full_steps.csv', index=False)
            if step % 10000 == 0 and batch != 0:
                val_loss.reset_states()
                for (batch, (inp, tar)) in enumerate(val_dataset.take(parameters['val_steps_per_epoch'])):
                    batch_start = time.time()
                    validation_step(inp, tar, encoder, decoder, tar_word_index, parameters['batch_size'], hidden)
                    batch_end = time.time()
                    if batch % 10 == 0:
                        print('Training step=' + str(step) + ', Batch=' + str(batch) + ', Validation Loss=' +
                              str(round(val_loss.result().numpy(), 3)) + ', Time taken=' +
                              str(round(batch_end - batch_start, 3)) + ' sec')
                d = {'steps': int(step), 'train_loss': train_loss.result().numpy(),
                     'val_loss': val_loss.result().numpy()}
                split_df = split_df.append(d, ignore_index=True)
                split_df.to_csv(loc_to+'history/split_steps.csv', index=False)
                print()
                print('Training step=' + str(step) + ', Training Loss=' + str(round(train_loss.result().numpy(), 3)) +
                      ', Validation Loss=' + str(round(val_loss.result().numpy(), 3)))
                if best_val_loss is None:
                    checkpoint_count = 0
                    best_val_loss = round(val_loss.result().numpy(), 3)
                    manager.save()
                    print('Checkpoint saved')
                    print()
                elif best_val_loss > round(val_loss.result().numpy(), 3):
                    checkpoint_count = 0
                    print('Best Validation Loss changed from ' + str(best_val_loss) + ' to ' +
                          str(round(val_loss.result().numpy(), 3)))
                    best_val_loss = round(val_loss.result().numpy(), 3)
                    manager.save()
                    print('Checkpoint saved')
                    print()
                elif checkpoint_count <= 10:
                    checkpoint_count += 1
                    print('Best Validation Loss did not improve')
                    print('Checkpoint not saved')
                    print()
                else:
                    print('Model did not improve after 10th time. Model stopped from training further.')
                    print()
                    step_break = True
                    break
                train_loss.reset_states()
            step += 1
            if step == parameters['max_train_steps']:
                print('Model finished training for ' + str(step))
                manager.save()
                print('Checkpoint saved')
                print()
                step_break = True
                break
        if step_break:
            break

def model_testing(test_dataset):
    global val_loss
    loc_to = '/home/preetham/Documents/neural-machine-translation/models/results/en-es/model_1/'
    val_loss = tf.keras.metrics.Mean(name='val_loss')
    parameters = open_file('parameters')
    val_loss.reset_states()
    encoder = Encoder(parameters['emb_size'], parameters['inp_vocab_size'], parameters['rnn_size'], parameters['rate'])
    decoder = Decoder(parameters['emb_size'], parameters['tar_vocab_size'], parameters['rnn_size'], parameters['rate'])
    checkpoint_dir = loc_to+'training_checkpoints'
    checkpoint = tf.train.Checkpoint(encoder=encoder, decoder=decoder)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    tar_word_index = open_file('tar-word-index')
    hidden = encoder.initialize_hidden_state(parameters['batch_size'], parameters['rnn_size'])
    for (batch, (inp, tar)) in enumerate(test_dataset.take(parameters['test_steps'])):
            validation_step(inp, tar, encoder, decoder, tar_word_index, parameters['batch_size'], hidden)
    print('Test Loss=', round(val_loss.result().numpy(), 3))
    print()

def preprocess_sentence(w):
    w = w.lower().strip()
    w = re.sub(r'[" "]+', " ", w)
    w = re.sub(r"[^a-zA-Z0-9.,!?;:']+", " ", w)
    w = re.sub(r'\s+', ' ', w)
    w = w.replace('.', ' . ')
    w = w.replace(',', ' , ')
    w = w.replace('!', ' ! ')
    w = w.replace('?', ' ? ')
    w = w.replace(':', ' : ')
    w = w.replace(';', ' ; ')
    w = w.replace("'", " ' ")
    w = w.strip()
    w = re.sub(r'\s+', ' ', w)
    return w

def word_separator(word):
    word = list(word)
    word = ['@'+word[0]]+['#'+i for i in word[1:-1]]+['$'+word[-1]]
    word = ' '.join(word)
    return word

def english_to_spanish(sentence):
    loc_to = '/home/preetham/Documents/neural-machine-translation/models/results/en-es/model_1/tokenizer/'
    inp_word_index = open_file('inp-word-index')
    sp = spm.SentencePieceProcessor()
    sp.Load(loc_to+'en.model')
    sentence = sp.EncodeAsPieces(sentence)
    tar_index_word = open_file('tar-index-word')
    tar_word_index = open_file('tar-word-index')
    parameters = open_file('parameters')
    sequence = [[inp_word_index[i] for i in sentence]]
    sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=parameters['inp_max_length'],
                                                             padding='post')
    sequence = tf.convert_to_tensor(sequence)
    english = []
    encoder = Encoder(parameters['emb_size'], parameters['inp_vocab_size'], parameters['rnn_size'], parameters['rate'])
    decoder = Decoder(parameters['emb_size'], parameters['tar_vocab_size'], parameters['rnn_size'], parameters['rate'])
    checkpoint_dir = '/home/preetham/Documents/neural-machine-translation/models/results/en-es/model_1/training_checkpoints'
    checkpoint = tf.train.Checkpoint(encoder=encoder, decoder=decoder)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    hidden = encoder.initialize_hidden_state(1, parameters['rnn_size'])
    enc_out, enc_hidden = encoder(sequence, False, hidden)
    dec_hidden = enc_hidden
    dec_inp = tf.expand_dims([tar_word_index['<s>']], 0)
    for i in range(1, parameters['tar_max_length']):
        prediction, dec_hidden = decoder(dec_inp, dec_hidden, enc_out, False)
        predicted_id = tf.argmax(prediction[0]).numpy()
        if tar_index_word[predicted_id] != '</s>':
            english.append(tar_index_word[predicted_id])
        else:
            break
        dec_inp = tf.expand_dims([predicted_id], 0)
    sp.Load(loc_to+'es.model')
    sentence = sp.DecodePieces(english)
    sentence = sentence.replace('▁', ' ')
    return sentence

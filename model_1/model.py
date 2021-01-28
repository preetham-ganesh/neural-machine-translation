import tensorflow as tf

class Encoder(tf.keras.Model):
    def __init__(self, emb_size, inp_vocab_size, rnn_size, rate):
        super(Encoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(inp_vocab_size, emb_size)
        self.rnn_fwd = tf.keras.layers.LSTM(int(rnn_size//2), return_state=True, return_sequences=True)
        self.rnn_bwd = tf.keras.layers.LSTM(int(rnn_size//2), return_state=True, return_sequences=True,
                                            go_backwards=True)
        self.bi_rnn = tf.keras.layers.Bidirectional(self.rnn_fwd, backward_layer=self.rnn_bwd)
        self.rnn_3 = tf.keras.layers.LSTM(rnn_size, return_state=True, return_sequences=True)
        self.rnn_4 = tf.keras.layers.LSTM(rnn_size, return_state=True, return_sequences=True)
        self.dropout = tf.keras.layers.Dropout(rate=rate)

    def call(self, inp, training, hidden):
        inp = self.embedding(inp)
        out, fwd_h, fwd_c, bwd_h, bwd_c = self.bi_rnn(inp, initial_state=hidden)
        out = self.dropout(out, training=training)
        h = tf.concat([fwd_h, bwd_h], axis=1)
        del fwd_h, bwd_h
        c = tf.concat([fwd_c, bwd_c], axis=1)
        del fwd_c, bwd_c
        out, h, c = self.rnn_3(out, initial_state=[h, c])
        out = self.dropout(out, training=training)
        out, h, c = self.rnn_4(out, initial_state=[h, c])
        out = self.dropout(out, training=training)
        return out, [h, c]

    def initialize_hidden_state(self, batch_size, units):
        fwd_h = tf.zeros((batch_size, int(units//2)))
        bwd_h = tf.zeros((batch_size, int(units//2)))
        fwd_c = tf.zeros((batch_size, int(units//2)))
        bwd_c = tf.zeros((batch_size, int(units//2)))
        return [fwd_h, fwd_c, bwd_h, bwd_c]

class LuongAttention(tf.keras.Model):
    def __init__(self, rnn_size):
        super(LuongAttention, self).__init__()
        self.wa = tf.keras.layers.Dense(rnn_size)

    def call(self, dec_out, enc_out):
        score = tf.matmul(dec_out, self.wa(enc_out), transpose_b=True)
        alignment = tf.nn.softmax(score, axis=2)
        context = tf.matmul(alignment, enc_out)
        return context

class Decoder(tf.keras.Model):
    def __init__(self, emb_size, tar_vocab_size, rnn_size, rate):
        super(Decoder, self).__init__()
        self.attention = LuongAttention(rnn_size)
        self.embedding = tf.keras.layers.Embedding(tar_vocab_size, emb_size)
        self.rnn_1 = tf.keras.layers.LSTM(rnn_size, return_state=True, return_sequences=True)
        self.rnn_2 = tf.keras.layers.LSTM(rnn_size, return_state=True, return_sequences=True)
        self.rnn_3 = tf.keras.layers.LSTM(rnn_size, return_state=True, return_sequences=True)
        self.rnn_4 = tf.keras.layers.LSTM(rnn_size, return_state=True, return_sequences=True)
        self.wc = tf.keras.layers.Dense(rnn_size, activation='tanh')
        self.dropout = tf.keras.layers.Dropout(rate=rate)
        self.ws = tf.keras.layers.Dense(tar_vocab_size)

    def call(self, inp, hidden, enc_out, training):
        inp = self.embedding(inp)
        out, h, c = self.rnn_1(inp, initial_state=hidden)
        out = self.dropout(out, training=training)
        out, h, c = self.rnn_2(out, initial_state=[h, c])
        out = self.dropout(out, training=training)
        out_, h_, c_ = self.rnn_3(out, initial_state=[h, c])
        out_ = self.dropout(out_, training=training)
        out = out + out_
        h = h + h_
        c = c + c_
        del out_, h_, c_
        out, h, c = self.rnn_4(out, initial_state=[h, c])
        out = self.dropout(out, training=training)
        context = self.attention(out, enc_out)
        out = tf.concat([tf.squeeze(context, 1), tf.squeeze(out, 1)], 1)
        out = self.wc(out)
        out = self.dropout(out, training=training)
        out = self.ws(out)
        return out, [h, c]
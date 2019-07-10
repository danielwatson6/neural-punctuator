"""Word-level seq2seq model with attention."""

import os

import editdistance
from gensim.models import KeyedVectors
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfkl

import boilerplate as tfbp


def Attention(input_size, hidden_size, style="bahdanau"):
    """Attention mechanism."""
    queries = tfkl.Input(shape=tf.TensorShape([None, input_size]))
    encoded = tfkl.Input(shape=tf.TensorShape([None, hidden_size]))

    h = tf.expand_dims(queries, 2)
    if style == "bahdanau":
        W1 = tfkl.Dense(hidden_size)
        W2 = tfkl.Dense(hidden_size)
        V = tfkl.Dense(1, use_bias=False)

        score = V(tf.nn.tanh(W1(encoded) + W2(h)))
    else:
        W = tfkl.Dense(hidden_size, use_bias=False)
        score = encoded @ W(h)

    result = tf.reduce_sum(tf.nn.softmax(score, axis=2) * encoded, axis=2)
    return tf.keras.Model(inputs=[queries, encoded], outputs=result)


def Decoder(
    vocab_size,
    embed_size,
    hidden_size,
    num_layers,
    attention="bahdanau",
    attention_pos=0,
    dropout=0.0,
):
    """Attention-based GRU decoder."""
    decoder_in = tfkl.Input(shape=tf.TensorShape([None, embed_size]))
    # The encoder is bidirectional and the left and right outputs are concatenated.
    encoder_out = tfkl.Input(shape=tf.TensorShape([None, 2 * hidden_size]))

    initial_state_dense = tfkl.Dense(num_layers * hidden_size)
    gru_layers = [
        tfkl.GRU(hidden_size, dropout=dropout, return_sequences=True)
        for _ in range(num_layers)
    ]
    softmax_dense = tfkl.TimeDistributed(tfkl.Dense(vocab_size))

    h0 = tf.reshape(encoder_out[:, 0, :], [-1, hidden_size])
    initial_states = tf.reshape(initial_state_dense(h0), [-1, num_layers, hidden_size])
    initial_states = tf.unstack(initial_states, num_layers, axis=1)

    dec_in = decoder_in
    for i in range(num_layers + 1):
        if i == attention_pos:
            attn = Attention(dec_in.shape[2], hidden_size, style=attention)(
                [dec_in, encoder_out]
            )
            dec_in = tf.concat([attn, dec_in], 2)
        if i < num_layers:
            dec_in = gru_layers[i](dec_in, initial_state=initial_states[i])

    probs = softmax_dense(dec_in)
    return tf.keras.Model(inputs=[decoder_in, encoder_out], outputs=probs)


@tfbp.default_export
class Seq2Seq(tfbp.Model):
    default_hparams = {
        "rnn_layers": 2,
        "batch_size": 32,
        "vocab_size": 20000,
        "hidden_size": 256,
        "attention": "bahdanau",  # "bahdanau" or "luong"
        "attention_pos": 0,
        "optimizer": "sgd",  # "sgd" or "adam"
        "learning_rate": 0.1,
        "num_valid": 1024,  # TODO: choose a good value
        "epochs": 20,
        "dropout": 0.0,
        "beam_width": 5,
        "corpus": 2,  # 2 or 103
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.step = tf.Variable(0, trainable=False)
        self.epoch = tf.Variable(0, trainable=False)

        self.embed = self._make_embed()

        self.encoder = tf.keras.Sequential()
        for _ in range(self.hparams.rnn_layers):
            self.encoder.add(tfkl.Bidirectional(self._make_gru()))

        self.decoder = Decoder(
            self.hparams.vocab_size,
            300,
            self.hparams.hidden_size,
            self.hparams.rnn_layers,
            self.hparams.attention,
            self.hparams.attention_pos,
            self.hparams.dropout,
        )

    def _make_embed(self):
        # Embedding matrix. TODO: move data-dependent stuff to data loader.
        word2vec = KeyedVectors.load(os.path.join("data", "word2vec"), mmap="r")
        embedding_matrix = np.random.uniform(
            low=-1.0, high=1.0, size=(self.hparams.vocab_size, 300)
        )
        corpus = self.hparams.corpus
        if corpus == 103:
            corpus = str(corpus) + "-raw"
        with open(os.path.join("data", f"wikitext-{corpus}", "wiki.vocab.tsv")) as f:
            for i, word in enumerate(f):
                word = word.strip()
                if word in word2vec:
                    embedding_matrix[i] = word2vec[word]
        return tfkl.Embedding(
            self.hparams.vocab_size,
            300,
            embeddings_initializer=tf.initializers.constant(embedding_matrix),
        )

    def _make_gru(self):
        dropout = 0.0
        if self.method == "fit":
            dropout = self.hparams.dropout

        return tfkl.GRU(
            self.hparams.hidden_size, dropout=dropout, return_sequences=True
        )

    def call(self, x):
        y = None
        if self.method == "fit":
            x, y = x

        encoder_inputs = self.embed(x)
        encoded = self.encoder(encoder_inputs)

        decoded = None
        if self.method == "fit":
            # Teacher forcing: prepend a <sos> token to the start of every sequence.
            decoder_in = self.embed(tf.concat([[[2]] * y.shape[0], y], 1))
            # Choose a decoder initial state.
            decoded = self.decoder([decoder_in, encoded])
        else:
            # TODO:
            ...

        return decoded

    def fit(self, data_loader):
        """Method invoked by `run.py`."""

        # Loss function.
        loss_fn = tf.losses.CategoricalCrossentropy(reduction=tf.losses.Reduction.NONE)

        # Optimizer.
        if self.hparams.optimizer == "adam":
            opt = tf.optimizers.Adam(self.hparams.learning_rate)
        else:
            opt = tf.optimizers.SGD(self.hparams.learning_rate)

        # Train/validation split. Keep a copy of the original validation data to
        # evalute at the end of every epoch without falling to an infinite loop.
        train_dataset, valid_dataset_orig = data_loader()
        valid_dataset = valid_dataset_orig.repeat()

        # TensorBoard writers.
        train_writer = tf.summary.create_file_writer(
            os.path.join(self.save_dir, "train")
        )
        valid_writer = tf.summary.create_file_writer(
            os.path.join(self.save_dir, "valid")
        )

        max_eval_score = float("-inf")

        while self.epoch.numpy() < self.hparams.epochs:
            for x, y in train_dataset:

                with tf.GradientTape() as tape:
                    train_loss = loss_fn(y, self([x, y]))
                grads = tape.gradient(train_loss, self.trainable_weights)
                opt.apply_gradients(zip(grads, self.trainable_weights))

                step = self.step.numpy()
                if step % 100 == 0:
                    x, y = next(valid_dataset)
                    valid_loss = loss_fn(y, self([x, y]))

                    with train_writer.as_default():
                        tf.summary.scalar(self.hparams.loss, train_loss, step=step)
                    with valid_writer.as_default():
                        tf.summary.scalar(self.hparams.loss, valid_loss, step=step)

                print("Step {} (train_loss={:.4f})".format(step, train_loss))
                self.step.assign_add(1)

            print(f"Epoch {self.epoch.numpy()} finished")
            self.epoch.assign_add(1)

            eval_score = self._evaluate(valid_dataset_orig, data_loader.id_to_word)
            if eval_score > max_eval_score:
                self.save()

    def _predict(self, x):
        """Beam search based output for input sequences."""
        seq_lengths = tf.reduce_sum(tf.cast(tf.math.not_equal(x, 0), tf.int64), axis=1)
        return tf.keras.backend.ctc_decode(
            self(x),
            seq_lengths,
            greedy=(self.hparams.beam_width == 1),
            beam_width=self.hparams.beam_width,
        )

    def _evaluate(self, dataset, id_to_word):
        """Levenshtein distance evaluation."""
        scores = []
        for x, y in dataset:
            y_sys = id_to_word(self._predict(x)).numpy()
            y = id_to_word(y).numpy()
            for pred, gold in zip(y_sys, y):
                scores.append(editdistance.eval(y_sys, y) / len(y))
        return sum(scores) / len(scores)

    def evaluate(self, data_loader):
        """Method invoked by `run.py`."""
        dataset = data_loader()
        return self._evaluate(dataset, data_loader.id_to_word)

    def interact(self, data_loader):
        """Method invoked by `run.py`."""
        print("Press Ctrl+C to quit.\n")
        dataset = data_loader()
        for x in dataset:
            y = self._predict(x)
            y = data_loader.id_to_word(y)
            print(y + "\n")

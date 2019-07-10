"""Word-level seq2seq model with attention."""

import os
import re

import editdistance
from gensim.models import KeyedVectors
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfkl

import boilerplate as tfbp


class Attention(tfkl.Layer):
    """Attention mechanism."""

    def __init__(self, hidden_size, style="bahdanau", **kwargs):
        super().__init__(**kwargs)
        self.style = style
        if style == "bahdanau":
            self.W1 = tfkl.Dense(hidden_size)
            self.W2 = tfkl.Dense(hidden_size)
            self.V = tfkl.Dense(1, use_bias=False)
        else:
            self.W = tfkl.Dense(hidden_size, use_bias=False)

    def call(self, x):
        h, e = x
        h = tf.expand_dims(h, 2)
        e = tf.expand_dims(e, 1)

        if self.style == "bahdanau":
            score = self.V(tf.math.tanh(self.W1(e) + self.W2(h)))
        else:
            score = e @ self.W(h)

        return tf.reduce_sum(tf.nn.softmax(score, axis=2) * e, axis=2)


class Decoder(tfkl.Layer):
    """Attention-based GRU decoder."""

    def __init__(
        self,
        vocab_size,
        hidden_size,
        num_layers,
        attention="bahdanau",
        attention_pos=0,
        dropout=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention_pos = attention_pos

        self.grus = [
            tfkl.GRU(
                hidden_size, dropout=dropout, return_sequences=True, return_state=True
            )
            for _ in range(num_layers)
        ]
        self.attention = Attention(hidden_size, style=attention)
        self.W2 = tfkl.TimeDistributed(tfkl.Dense(vocab_size, activation=tf.nn.softmax))

    def call(self, x):
        dec_in, dec_st, enc_out = x
        new_dec_st = []

        for i in range(self.num_layers + 1):
            if i == self.attention_pos:
                dec_in = tf.concat([self.attention([dec_in, enc_out]), dec_in], 2)
            if i < self.num_layers:
                dec_in, st = self.grus[i](dec_in, initial_state=dec_st[i])
                new_dec_st.append(st)

        return self.W2(dec_in), new_dec_st


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

        dropout = 0.0
        if self.method == "fit":
            dropout = self.hparams.dropout

        self.encoder = tf.keras.Sequential()
        for _ in range(self.hparams.rnn_layers):
            self.encoder.add(
                tfkl.Bidirectional(
                    tfkl.GRU(
                        self.hparams.hidden_size, dropout=dropout, return_sequences=True
                    )
                )
            )

        # Initial states for the decoder are outputted by W1(enc_out[0]).
        self.decoder_initial_state_dense = tfkl.Dense(
            self.hparams.rnn_layers * self.hparams.hidden_size, activation=tf.math.tanh
        )

        self.decoder = Decoder(
            self.hparams.vocab_size,
            self.hparams.hidden_size,
            self.hparams.rnn_layers,
            attention=self.hparams.attention,
            attention_pos=self.hparams.attention_pos,
            dropout=dropout,
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

    def _loss(self, y_true, y_pred):
        mask = tf.cast(tf.math.not_equal(y_true, 0), tf.float32)
        seq_lengths = tf.expand_dims(tf.reduce_sum(mask, axis=1), 1)
        ce = tf.losses.SparseCategoricalCrossentropy(reduction=tf.losses.Reduction.NONE)
        result = ce(y_true, y_pred, sample_weight=mask)
        return tf.reduce_mean(result / seq_lengths)

    def call(self, x):
        y = None
        if self.method == "fit":
            x, y = x

        enc_in = self.embed(x)
        enc_out = self.encoder(enc_in)

        # Decoder initial state and input.
        dec_st = self.decoder_initial_state_dense(enc_out[:, 0, :])
        dec_st = tf.reshape(
            dec_st, [-1, self.hparams.rnn_layers, self.hparams.hidden_size]
        )
        dec_st = tf.unstack(dec_st, self.hparams.rnn_layers, axis=1)

        sos_ids = tf.cast([[2]] * x.shape[0], tf.int64)

        if self.method == "fit":
            # Teacher forcing: prepend a <sos> token to the start of every sequence.
            dec_in = self.embed(tf.concat([sos_ids, y[:, :-1]], 1))
            dec_out, _ = self.decoder([dec_in, dec_st, enc_out])

        else:
            dec_in = self.embed(sos_ids)
            dec_out = []

            # Give some extra space for decoding. TODO: generalize to other tasks.
            for _ in range(x.shape[1] + 20):
                out, dec_st = self.decoder([dec_in, dec_st, enc_out])
                # Greedy decoding: next input = embed of max likelihood output token.
                next_token = tf.argmax(out, axis=-1)
                dec_in = self.embed(next_token)
                dec_out.append(out)

            dec_out = tf.squeeze(tf.stack(dec_out, axis=1), 2)

        return dec_out

    def fit(self, data_loader):
        """Method invoked by `run.py`."""

        # Optimizer.
        if self.hparams.optimizer == "adam":
            opt = tf.optimizers.Adam(self.hparams.learning_rate)
        else:
            opt = tf.optimizers.SGD(self.hparams.learning_rate)

        # Train/validation split. Keep a copy of the original validation data to
        # evalute at the end of every epoch without falling to an infinite loop.
        train_dataset, valid_dataset_orig = data_loader()
        valid_dataset = iter(valid_dataset_orig)

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

                # TODO: try to remove this
                if not self.built:
                    self([x, y])

                with tf.GradientTape() as g:
                    train_loss = self._loss(y, self([x, y]))

                grads = g.gradient(train_loss, self.trainable_weights)
                opt.apply_gradients(zip(grads, self.trainable_weights))

                step = self.step.numpy()
                if step % 100 == 0:
                    x, y = next(valid_dataset)
                    valid_loss = self._loss(y, self([x, y]))

                    with train_writer.as_default():
                        tf.summary.scalar("cross_entropy", train_loss, step=step)
                    with valid_writer.as_default():
                        tf.summary.scalar("cross_entropy", valid_loss, step=step)

                print("Step {} (train_loss={:.4f})".format(step, train_loss))
                self.step.assign_add(1)

            print(f"Epoch {self.epoch.numpy()} finished")
            self.epoch.assign_add(1)

            eval_score = self._evaluate(
                valid_dataset_orig, data_loader.id_to_word.lookup
            )
            if eval_score > max_eval_score:
                self.save()
                with valid_writer.as_default():
                    tf.summary.scalar("edit_distance", eval_score, step=step)

    def _predict(self, x):
        """Beam search based output for input sequences."""
        y = self(x)
        seq_lengths = tf.expand_dims(y.shape[1], 0)
        return tf.keras.backend.ctc_decode(
            y,
            seq_lengths,
            greedy=(self.hparams.beam_width == 1),
            beam_width=self.hparams.beam_width,
        )[0]

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
        return self._evaluate(dataset, data_loader.id_to_word.lookup)

    def interact(self, data_loader):
        """Method invoked by `run.py`."""
        dataset = data_loader()
        for x in dataset:
            y = data_loader.id_to_word.lookup(self._predict(x)[0][0]).numpy()
            y = " ".join([token.decode("utf-8") for token in y])
            y = re.sub(r" <eos>.*", "", y)
            print("Output sentence:", y + "\n")

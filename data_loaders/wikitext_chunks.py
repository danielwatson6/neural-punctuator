"""Wikitext data loader.

The following special word-level tokens are introduced:
    <pad>: used for batch processing of variable-length sentences
    <unk>: unknown, out-of-vocabulary tokens
    <sos>: start-of-sentence token
    <eos>: end-of-sentence token
    <dash>: dash inside a compound word
    <num>: a number
    <num_dot>: a decimal separator inside a number
    <num_comma>: a comma separator inside a number

The following run.py methods are compatible with this data loader:
    fit
    evaluate
    interact

"""

import os

import tensorflow as tf

import boilerplate as tfbp


@tfbp.default_export
class WikiText(tfbp.DataLoader):
    default_hparams = {"batch_size": 32, "corpus": 2, "seq_len": 40}

    def call(self):
        if self.hparams.corpus == 2:
            data_path = os.path.join("data", "wikitext-103-raw")
        elif self.hparams.corpus == 103:
            data_path = os.path.join("data", "wikitext-2")
        else:
            raise ValueError("`corpus` hyperparameter can only attain values 2 or 103.")

        vocab_path = os.path.join(data_path, "wiki.vocab.tsv")

        # Args: filename, key_dtype, key_index, value_dtype, value_index, vocab_size
        word_to_id_init = tf.lookup.TextFileInitializer(
            vocab_path,
            tf.string,
            0,
            tf.int64,
            tf.lookup.TextFileIndex.LINE_NUMBER,
            vocab_size=self.hparams.vocab_size,
        )
        id_to_word_init = tf.lookup.TextFileInitializer(
            vocab_path,
            tf.int64,
            tf.lookup.TextFileIndex.LINE_NUMBER,
            tf.string,
            0,
            vocab_size=self.hparams.vocab_size,
        )
        self.word_to_id = tf.lookup.StaticHashTable(word_to_id_init, 1).lookup
        self.id_to_word = tf.lookup.StaticHashTable(id_to_word_init, "<unk>").lookup

        if self.method == "fit":
            train_inputs = self._create_dataset(
                os.path.join(data_path, "wiki.train.inputs")
            )
            train_labels = self._create_dataset(
                os.path.join(data_path, "wiki.train.labels")
            )
            valid_inputs = self._create_dataset(
                os.path.join(data_path, "wiki.valid.inputs")
            )
            valid_labels = self._create_dataset(
                os.path.join(data_path, "wiki.valid.labels")
            )
            train_dataset = tf.data.Dataset.zip((train_inputs, train_labels))
            valid_dataset = tf.data.Dataset.zip((valid_inputs, valid_labels))

            train_dataset = self._transform_dataset(train_dataset)
            valid_dataset = self._transform_dataset(valid_dataset)

            return train_dataset, valid_dataset

        elif self.method == "evaluate":
            test_inputs = tf.data.TextLineDataset(
                os.path.join(data_path, "wiki.test.inputs")
            )
            test_labels = tf.data.TextLineDataset(
                os.path.join(data_path, "wiki.test.labels")
            )
            test_dataset = tf.data.Dataset.zip((test_inputs, test_labels))

            return self._transform_dataset(test_dataset)

        elif self.method == "interact":

            def g():
                while True:
                    yield [input("Type a sentence: ")]

            dataset = tf.data.Dataset.from_generator(g, tf.string)
            return dataset.map(self.sent_to_id)

    def sent_to_id(self, x):
        x = tf.strings.split(x + " <eos>").to_tensor(default_value="<pad>")
        return self.word_to_id(x)

    def id_to_sent(self, x):
        x = tf.strings.join(self.id_to_word(x), separator=" ")
        # Remove " <eos>" and everything after.
        x = tf.strings.regex_replace(x, r"((?:[^ ]| [^<]| <[^e])*)(?: <e.*)?", r"\1")
        x = tf.strings.regex_replace(x, r" <dash> ", r"-")
        x = tf.strings.regex_replace(x, r" <num_dot> ", r"\.")
        return tf.strings.regex_replace(x, r" <num_comma> ", r",")

    def _create_dataset(self, path):

        def g():
            with open(path) as f:
                buf = []
                for line in f:
                    for word in line.split():
                        if len(buf) == self.hparams.seq_len:
                            yield " ".join(buf)
                            buf = []
                        buf.append(word)

        return tf.data.Dataset.from_generator(g, tf.string)

    def _transform_dataset(self, dataset):
        dataset = dataset.batch(self.hparams.batch_size)
        dataset = dataset.map(lambda x, y: (self.sent_to_id(x), self.sent_to_id(y)))
        return dataset.prefetch(1)

"""Word-level wikitext data loader.

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
    default_hparams = {
        "batch_size": 32,
        "corpus": 2,
        "max_seq_len": 40,
        "vocab_size": 20000,
        "shuffle": True,
        "chunk": False,
    }

    def call(self):
        if self.hparams.corpus == 2:
            data_path = os.path.join("data", "wikitext-103-raw")
        elif self.hparams.corpus == 103:
            data_path = os.path.join("data", "wikitext-2")
        else:
            raise ValueError("`corpus` hyperparameter can only attain values 2 or 103.")

        vocab_path = os.path.join(data_path, "wiki.vocabulary.tsv")

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
            train_labels = self._create_dataset(
                os.path.join(data_path, "wiki.train.clean")
            )
            valid_labels = self._create_dataset(
                os.path.join(data_path, "wiki.valid.clean")
            )
            return (
                self._transform_dataset(train_labels),
                self._transform_dataset(valid_labels),
            )

        elif self.method == "evaluate":
            test_labels = self._create_dataset(
                os.path.join(data_path, "wiki.test.clean")
            )
            return self._transform_dataset(test_labels)

        elif self.method == "interact":

            def g():
                while True:
                    yield [input("Type a sentence: ")]

            dataset = tf.data.Dataset.from_generator(g, tf.string)
            return dataset.map(self.sent_to_id)

    def sent_to_id(self, x):
        x = tf.strings.split(x + " <eos>").to_tensor(default_value="<pad>")
        if self.method == "train" and not self.hparams.chunk:
            x = x[:, : self.hparams.max_seq_len]
        return self.word_to_id(x)

    def id_to_sent(self, x):
        x = tf.strings.join(self.id_to_word(x), separator=" ")
        # Remove " <eos>" and everything after.
        x = tf.strings.regex_replace(x, r"((?:[^ ]| [^<]| <[^e])*)(?: <e.*)?", r"\1")
        x = tf.strings.regex_replace(x, r" <dash> ", r"-")
        x = tf.strings.regex_replace(x, r" <num_dot> ", r"\.")
        return tf.strings.regex_replace(x, r" <num_comma> ", r",")

    def _create_dataset(self, path):
        if not self.hparams.chunk:
            return tf.data.TextLineDataset(path)

        def g():
            with open(path) as f:
                buf = []
                for line in f:
                    for word in line.split():
                        if len(buf) == self.hparams.max_seq_len:
                            yield " ".join(buf)
                            buf = []
                        buf.append(word)

        return tf.data.Dataset.from_generator(g, tf.string)

    def _make_inputs(self, y):
        x = tf.strings.regex_replace(y, r"\.;", r" ")
        x = tf.strings.regex_replace(x, r"\s+", r" ")
        x = tf.strings.strip(x)
        return self.sent_to_id(x), self.sent_to_id(y)

    def _transform_dataset(self, dataset):
        if self.hparams.shuffle:
            dataset = dataset.shuffle(10000)
        dataset = dataset.batch(self.hparams.batch_size)
        dataset = dataset.map(self._make_inputs)
        return dataset.prefetch(1)

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
import re

import tensorflow as tf

import boilerplate as tfbp


@tfbp.default_export
class WikiText(tfbp.DataLoader):
    default_hparams = {
        "batch_size": 32,
        "corpus": 2,
        "max_seq_len": 400,
        "vocab_size": 128,
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

        vocab_path = os.path.join(data_path, "wiki.alphabet.tsv")

        # Args: filename, key_dtype, key_index, value_dtype, value_index, vocab_size
        char_to_id_init = tf.lookup.TextFileInitializer(
            vocab_path,
            tf.string,
            0,
            tf.int64,
            tf.lookup.TextFileIndex.LINE_NUMBER,
            vocab_size=self.hparams.vocab_size,
        )
        id_to_char_init = tf.lookup.TextFileInitializer(
            vocab_path,
            tf.int64,
            tf.lookup.TextFileIndex.LINE_NUMBER,
            tf.string,
            0,
            vocab_size=self.hparams.vocab_size,
        )
        self.char_to_id = tf.lookup.StaticHashTable(char_to_id_init, 1).lookup
        self.id_to_char = tf.lookup.StaticHashTable(id_to_char_init, "<unk>").lookup

        if self.method == "fit":
            train_labels = tf.data.TextLineDataset(
                os.path.join(data_path, "wiki.train.clean")
            )
            valid_labels = tf.data.TextLineDataset(
                os.path.join(data_path, "wiki.valid.clean")
            )
            return (
                self._transform_dataset(train_labels),
                self._transform_dataset(valid_labels),
            )

        elif self.method == "evaluate":
            test_labels = tf.data.TextLineDataset(
                os.path.join(data_path, "wiki.test.clean")
            )
            return self._transform_dataset(test_labels)

        elif self.method == "interact":

            def interact_mode_generator():
                while True:
                    yield [input("Type a sentence: ")]

            dataset = tf.data.Dataset.from_generator(interact_mode_generator, tf.string)
            return dataset.map(self.sent_to_id)

    def sent_to_id(self, x):
        # Don't split special tokens into characters. To do this, we separate everything
        # except special tokens with tabs, and then split by tabs.
        x = tf.strings.regex_replace(x + "<eos>", r"(<[^>]+>|[^<])", r"\1\t")
        x = tf.strings.split(x, sep="\t").to_tensor(default_value="<pad>")

        if self.method == "train" and not self.hparams.chunk:
            x = x[:, : self.hparams.max_seq_len]
        return self.char_to_id(x)

    def id_to_sent(self, x):
        x = tf.strings.join(self.id_to_char(x))
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
                    for token in re.sub(r"(<[^>]+>|[^<])", r"\1\t", line).split("\t"):
                        if len(buf) == self.hparams.max_seq_len:
                            yield "".join(buf)
                            buf = []
                        buf.append(token)

        return tf.data.Dataset.from_generator(g, tf.string)

    def _make_inputs(self, y):
        x = tf.strings.regex_replace(y, r"\.;", r" ")
        x = tf.strings.regex_replace(x, r"\s+", r" ")
        x = tf.strings.strip(x)
        return self.sent_to_id(x), self.sent_to_id(y)

    def _transform_dataset(self, dataset):
        if self.hparams.shuffle and not self.hparams.chunk:
            dataset = dataset.shuffle(10000)
        dataset = dataset.batch(self.hparams.batch_size)
        dataset = dataset.map(self._make_inputs)
        return dataset.prefetch(1)

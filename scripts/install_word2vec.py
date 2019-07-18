"""Script to rewrite Google's word2vec into a format that loads faster."""

import os

from gensim.models import KeyedVectors


if __name__ == "__main__":
    path = os.path.join("data", "GoogleNews-vectors-negative300.bin.gz")
    w2v = KeyedVectors.load_word2vec_format(path, binary=True)
    w2v.init_sims(replace=True)
    w2v.save(path)

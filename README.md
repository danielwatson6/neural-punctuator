# Neural Punctuator

Seq2Seq model that restores punctuation on English input text.

## Setup

Python 3.7.3 and TensorFlow 2.0 required.

For more information on the project structure, see the README in the [tensorflow-boilerplate](https://github.com/danielwatson6/tensorflow-boilerplate) repository.

### Datasets

- [WikiText](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/): download, unzip, and place both of the word-level datasets in the `data` directory. Clean the data with `python -m scripts.clean_wikitext`.

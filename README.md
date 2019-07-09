# Neural Punctuator

Seq2Seq model that restores punctuation on English input text.

## Setup

Python 3.7.3 and TensorFlow 2.0 are required to run this project.
```bash
virtualenv env
source env.sh
pip install -r requirements.txt
pip install tensorflow==2.0.0b1  # or tensorflow-gpu / path to a custom wheel
```

For more information on the project structure, see the README in the [tensorflow-boilerplate](https://github.com/danielwatson6/tensorflow-boilerplate) repository.

### Datasets

- [Google News Word2Vec](): place the .bin file in the `data` directory. Reformat the file into a format that loads faster by running `python -m scripts.install_word2vec`.

- [WikiText](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/): download, unzip, and place both of the word-level datasets in the `data` directory. Clean the data with `python -m scripts.clean_wikitext`.

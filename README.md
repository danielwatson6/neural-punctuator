# Neural Punctuator

Seq2Seq model that restores punctuation on English input text.

## Setup

Python 3.7.4, virtualenv and TensorFlow 2.0.0-beta1 are required to run this project.
```bash
virtualenv env
source env.sh
pip install -r requirements.txt
pip install tensorflow==2.0.0-beta1  # or tensorflow-gpu==2.0.0-beta1 / custom wheel
```

For more information on the project structure, see the README in the [tensorflow-boilerplate](https://github.com/danielwatson6/tensorflow-boilerplate) repository.

### Datasets

- [Google News Word2Vec](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing): place the .bin file in the `data` directory, run `python -m scripts.install_word2vec`, and optionally delete the `.bin` file.

- [WikiText](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/): download, unzip, and place both of the word-level datasets in the `data` directory. Clean the data with `python -m scripts.clean_wikitext`, and optionally delete the original `*.tokens` and `*.raw` files.

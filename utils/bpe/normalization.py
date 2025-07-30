import re

def text_normalization(corpus: str):
    corpus = corpus.lower()
    corpus = corpus.replace('\n', '')
    return re.sub(' +', '_', corpus)
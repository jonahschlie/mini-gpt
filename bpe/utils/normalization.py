import regex as re

def text_normalization(corpus: str):
    """
    Normalize a given text corpus by applying the following steps:

    1. Convert all characters to lowercase.
    2. Remove newline characters.
    3. Replace multiple consecutive spaces with a single underscore ('_').

    Args:
        corpus (str): The input text corpus to normalize.

    Returns:
        str: The normalized text corpus.
    """

    corpus = corpus.lower()
    corpus = corpus.replace('\n', '')
    corpus = re.sub(' +', '_', corpus)
    return re.sub(r'[\p{Han}\p{Hiragana}\p{Katakana}\p{Hangul}\p{Arabic}\p{Hebrew}]', '', corpus)
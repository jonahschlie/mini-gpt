def update_corpus(tokenized_corpus, first_char, second_char):
    """
    Find a given pair of letters within the corpus and replace them with a single merged string.
    Args:
        tokenized_corpus (list of str): A list of tokens (strings) to analyze.
        first_char (str): first string in target keyword (can be more than one character)
        second_char (str): second string in target keyword (can be more than one character)
    Returns:
        list of tuple: A list of ((token1, token2), count) tuples sorted by count.
    """
    index_to_delete = []
    for letter_index in range(len(tokenized_corpus) - 1):
        if tokenized_corpus[letter_index] == first_char and tokenized_corpus[letter_index + 1] == second_char:
            tokenized_corpus[letter_index] = first_char + second_char
            index_to_delete = index_to_delete + [letter_index + 1]
        else:
            continue
    index_to_delete.reverse()
    for index in index_to_delete:
        del tokenized_corpus[index]

    return tokenized_corpus
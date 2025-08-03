def update_corpus(tokenized_corpus, first_char, second_char):
    """
    Merge all occurrences of a given adjacent token pair into a single token
    within the tokenized corpus.

    Args:
        tokenized_corpus (list of str): A list of tokens to update.
        first_char (str): The first token in the target pair.
        second_char (str): The second token in the target pair.

    Returns:
        list of str: The updated tokenized corpus after merging the token pair.
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
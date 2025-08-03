def get_most_frequent_pair(tokenized_corpus):
    """
    Count and rank adjacent token pairs in a tokenized corpus and
    return the most frequent one. Skips token pairs where the
    first token ends with an underscore ('_').

    Args:
        tokenized_corpus (list of str): A list of tokens (strings) to analyze.

    Returns:
        tuple: The most frequent adjacent token pair as (token1, token2).
    """
    pair_dict = {}
    for letter in range(len(tokenized_corpus) - 1):
        if tokenized_corpus[letter][-1] == '_':
            continue
        else:
            pair = (tokenized_corpus[letter], tokenized_corpus[letter + 1])
            if pair in pair_dict:
                pair_dict[pair] = pair_dict[pair] + 1
            else:
                pair_dict[pair] = 1

    pair_ranking = sorted(pair_dict.items(), key=lambda item: item[1], reverse=True)
    return pair_ranking[0][0] + pair_ranking[0][0]
def extract_vocabulary(text: str):
    # use to get our starting set based on corpus
    vocabulary = set()  # using a set automatically filters duplicates
    for letter in text.lower():
        vocabulary.add(letter)
    return vocabulary
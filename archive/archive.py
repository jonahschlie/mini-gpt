'''
#model = NGramModel(n=2, lambdas=[1, 0], model_path="models/neural_ngram/bigram.json")
#model = NGramModel(n=2, lambdas=[0, 1], model_path="models/neural_ngram/bigram.json")
#model = NGramModel(n=3, lambdas=[0,0.4,0.6], model_path="models/neural_ngram/trigram.json")
#model = NGramModel(n=4, lambdas=[0.1,0.1,0.3,0.5], model_path="models/neural_ngram/fourgram.json")
#model = NGramModel(n=1, model_path="models/neural_ngram/unigram.json")

try:
    model.load()
    print('Loaded existing neural_ngram model')
except FileNotFoundError:
    model.fit(encoded_training_data)
    model.save()


print(model.calculate_perplexity(encoded_test_data))
'''


# generated = model.generate_sequence(length=30, seed=("be_",))
# print(re.sub('_', ' ', "".join(generated)))
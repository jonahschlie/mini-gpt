## What is this repository about?

This repository constitutes the final assigment for the course "Buildung GPT from scratch" taught by Prof. Elia Bruni and was submitted by Jonah Schlie, William Sholer, Zeynep Kocak and xxxx. 

During the course we investigated step by step how GPTs are build inclusive preprocessing, preceding language models before the modern Transformer Achitecture came up, and finally the transformer based mini GPT. The course and therefore the repo as well are structured in 4 milestones which were implemented by us. 
1. Byte Pair Encoder
2. Classical N-Gram
3. Neural N-Gram
4. GPT

## Repository Structure

The repository is mainly structured along the 4 mentioned milestones as you can see below:

```markdown
mini-gpt/  
├── data/              # shakespeare text data and modern corpus
├── bpe/               # Code for BPE (Milestone 1)
├── ngram/             # Classical N-Gram Model impelmentation
├── neural_ngram/      # Classical and Neural N-Gram Model impelmentation
├── gpt /              # The GPT implementation
├── utils/             # Functions that are used across all milestones  
├── requirements.txt/  # list of needed python packages to run the code 
├── main.py            # The whole GPT pipeline + test generation
└── README.md          # Technical Report
```


Each milestone folder contains the model implementation, a utils folder with general-purpose functions (kept separate from the model class for best practices), and a main script used for testing, optimization, and experimentation (details provided later).

## Getting Started

##### Installation  
```bash  
git clone https://github.com/your-username/mini-gpt.git
cd mini-gpt
pip install -r requirements.txt
```

##### Running
To run the main GPT script just use the following command in the repos root directory:
```bash
python -m main
```

To run the main script of each milestone independently use the following command in the repos root directory:
```bash
python -m bpe.main
python -m ngram.main
python -m gpt.main
```
What the actual scripts are about will be described detailed in the following technical report.

## Technical Report

### Unix Command



---
### Byte Pair Encoding
###### Byte Pair Encoder Class
We implemented the Byte Pair Encoder as a Python class which is structured as followed:

Constructor:

| Attribute  | Type | Description                                            |
| ---------- | ---- | ------------------------------------------------------ |
| vocab_size | int  | Number of desired merge operations                     |
| verbose    | bool | if True, detailed output is printet while "training"   |
| model_path | str  | Optional default path for saving and loading the model |
| neural     | bool | If True the model will return indicies after encoding  |

Functions

| Method            | Parameters                                | Returns | Description                                                                            |
| ----------------- | ----------------------------------------- | ------- | -------------------------------------------------------------------------------------- |
| fit               | corpus: str                               | None    | Fit the model on the given corpus.                                                     |
| encode            | corpus: str                               | list    | Encode the given corpus.                                                               |
| calculate_metrics | corpus: str, tokens: [str], verbose: bool | tuple   | Calculate ATL and TPW for the given corpus.                                            |
| save              | filepath: str                             | None    | Save model countings, vocab size, and stoi/itos dictionaries in JSON.                  |
| load              | filepath: str                             | None    | Load model countings, vocab size, and stoi/itos dictionaries from JSON for efficiency. |

###### Util Functions
For granularity why sources some of the used functionality out to static methods. By this we wanted to keep the code clean and maintainable:

*normalization.py* <br>
This function prepares to corpus before it gets tokenized to character level. The passed text is first converted to only lower cases, then all line breaks gets removed and last all spaces are replaced with _ which then are used as a end of word token so to say. The reason why we removed the line breaks was because the later models should focus on learning next words and generate text and not on text formation or where to use line breaks and so on.

*char_tokenization.py:* <br>
This function get a corpus and returns a list of all characters separated as a item of that list. This function is used by the Encoder firstly to create a basis of defining the initial vocabulary as well as preparing the text to be encoded for the encoding algorithm.

*get_most_frequent_token_pair.py* <br>
This function scans the tokenized corpus and identifies the most frequent adjacent token pair. It returns the token pair and its frequency, which is then used to determine which tokens should be merged in the next step of the Byte Pair Encoding algorithm.

*merge_token_pair_in_corpus.py* <br>
This function takes the most frequent token pair and merges all its occurrences in the corpus into a single token. This step updates the tokenized representation of the corpus and is repeated iteratively to grow the vocabulary according to the Byte Pair Encoding process.


| Method                       | Parameters                             | Returns | Description                                                                                                                                  |
| ---------------------------- | -------------------------------------- | ------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| text_normalization           | corpus: str                            | str     | Normalizes text by converting it to lowercase, removing newline characters, and replacing multiple spaces with a single underscore (`_`).    |
| char_tokenization            | corpus: str                            | list    | Tokenizes text into a list of individual characters.                                                                                         |
| get_most_frequent_token_pair | corpus: list of str                    | tuple   | Identifies and returns the most frequent adjacent token pair in the tokenized corpus.                                                        |
| merge_token_pair_in_corpus   | corpus: list of str, token_pair: tuple | list    | Merges all occurrences of the specified token pair into a single token within the tokenized corpus and returns the updated tokenized corpus. |

###### bpe/main.py
In the bpe/main.py file, we investigated two aspects related to the **generalization capability** of BPE using the **Tokens per Word (TPW)** metric.

 **1. Generalization Loss within Shakespeare Texts** <br>
First, we wanted to determine the value of **k** (the number of BPE merges) at which a tokenizer trained on the **training Shakespeare dataset** would start to lose its generalization when evaluated on a **test Shakespeare dataset**.

We used a **greedy-like approach** (maybe Pseudocode??):
- Incrementally increase k.
- Train a BPE tokenizer on the training data.
- Compute **TPW** for both training and test datasets for comparison.

The **scoring function** was designed to combine both tokenization efficiency and generalization:
  
$$
\text{score} = \text{test_tpw} *\frac{\text{test_tpw}}{\text{train_tpw}}
$$

We expected this score to remain high because Shakespeare’s works share a relatively consistent vocabulary and style.

- The tokens learned on the training data should apply well to the test data.
- Only at very high k values (where BPE effectively memorizes entire words) should generalization begin to decline, as it starts creating tokens specific to words seen only in training.

Results for this experiment are shown in the table below.

|   k   | train_tpw | test_tpw | generalization | score  |
|:-----:|:---------:|:--------:|:--------------:|:------:|
|   0   |  5.4521   |  5.4466  |     0.9990     | 5.4411 |
| 2000  |  1.5188   |  1.5155  |     0.9978     | 1.5121 |
| 4000  |  1.3316   |  1.3383  |     1.0050     | 1.3450 |
| 6000  |  1.2413   |  1.2589  |     1.0142     | 1.2767 |
| 8000  |  1.1870   |  1.2118  |     1.0209     | 1.2371 |
| 10000 |  1.1494   |  1.1809  |     1.0275     | 1.2134 |
| 12000 |  1.1237   |  1.1631  |     1.0351     | 1.2039 |
| 14000 |  1.0987   |  1.1436  |     1.0409     | 1.1904 |
| 16000 |  1.0842   |  1.1378  |     1.0495     | 1.1941 |
| 18000 |  1.0712   |  1.1310  |     1.0558     | 1.1940 |
| 20000 |  1.0585   |  1.1156  |     1.0539     | 1.1757 |
| 22000 |  1.0458   |  1.1081  |     1.0596     | 1.1741 |
| 24000 |  1.0330   |  1.1029  |     1.0677     | 1.1776 |
| 26000 |  1.0204   |  1.0962  |     1.0743     | 1.1776 |
| 28000 |  1.0078   |  1.0906  |     1.0822     | 1.1803 |

The table shows that the optimal number of BPE merges (**k**) is around **22,000** before generalization from the training dataset to the test dataset begins to degrade.

We can already observe that the generalization ratio exceeds **1.0** as early as **k = 4,000**, which means that not all tokens learned from the training data transfer perfectly to the test data. However, the **score** continues to improve because the test dataset’s **Tokens per Word (TPW)** decreases significantly, which compensates for the slight loss in generalization, exactly as described by the scoring function.

Only beyond **k > 22,000** does this trend reverse: too many tokens are specialized to the training data and cannot be applied effectively to the test data, resulting in no further score improvements. At this point, the training TPW (1.0458) indicates that BPE has effectively learned nearly every individual word in the training corpus. Consequently, the algorithm stops after three consecutive iterations without improvement and sets **k = 22,000** as the optimal value with respect to generalization.

We are fully aware that such a high **k** is unusual for a corpus of this size, and it is primarily due to the strong stylistic consistency and limited vocabulary variance across Shakespeare’s texts. For this reason, we repeated the experiment with a modern corpus as the test dataset to see how far the tokenizer could generalize to contemporary language. It is important to note that, in practice, one would not use Shakespearean text when building a language model intended exclusively for 

**2. Generalization Loss on Modern English (WikiText-2)** <br>
Secondly, we repeated the same experiment using a **modern English dataset (WikiText-2)** as the test set.

Here, we expected the **critical value of k (where generalization is lost)** to be much lower. The reason is that a tokenizer trained on Shakespearean English would produce tokens specialized for older language constructs, making it less effective on modern English text.

Results for this experiment are shown in the table below.

| k    | train_tpw | test_tpw | generalization | score  |
| ---- | --------- | -------- | -------------- | ------ |
| 0    | 5.4521    | 5.3059   | 0.9732         | 5.1636 |
| 100  | 3.0987    | 3.2343   | 1.0437         | 3.3757 |
| 200  | 2.6170    | 2.8828   | 1.1016         | 3.1756 |
| 300  | 2.3712    | 2.6794   | 1.1300         | 3.0277 |
| 400  | 2.2142    | 2.5775   | 1.1641         | 3.0005 |
| ...  | ...       | ...      | ...            | ...    |
| 3000 | 1.4032    | 1.9635   | 1.3993         | 2.7476 |
| 3100 | 1.3944    | 1.9563   | 1.4030         | 2.7447 |
| 3200 | 1.3863    | 1.9486   | 1.4057         | 2.7391 |
| 3300 | 1.3786    | 1.9410   | 1.4079         | 2.7328 |
| 3400 | 1.3712    | 1.9359   | 1.4118         | 2.7332 |
| 3500 | 1.3636    | 1.9314   | 1.4164         | 2.7356 |
| 3600 | 1.3565    | 1.9260   | 1.4199         | 2.7347 |

The results clearly show that the optimal number of BPE merges (k) when evaluated on modern English is significantly lower than for Shakespearean English.

Here, generalization starts to degrade much earlier, with the generalization ratio already exceeding 1.1 around k = 200 and continuing to grow beyond k = 3,000. This indicates that a tokenizer trained specifically on Shakespeare’s older vocabulary and phrasing does not generalize as well to contemporary English, which has different word constructions, syntax, and vocabulary.

Despite this, the overall score still decreases as k increases up to a point because the test TPW improves. However, once the loss in generalization outweighs tokenization gains, the score stabilizes and eventually increases slightly.

This behavior shows that the optimal trade-off for generalization and tokenization efficiency on modern English is reached at a much smaller k (≈ 3,000) compared to the Shakespeare-to-Shakespeare experiment (≈ 22,000). It also illustrates how domain mismatch between training and testing corpora can significantly impact tokenizer performance.

---
### Classical N-Gram
###### NGRAM Class Implementation
The NGramModel class builds and uses an n-gram language model that learns word sequence probabilities from training data. It supports interpolation across multiple n-gram orders, allows predicting next words, generating sequences, calculating perplexity, and saving/loading trained models.

**Constructor**

| Attribute  | Type | Description                                                            |
| ---------- | ---- | ---------------------------------------------------------------------- |
| n          | int  | Maximum order of n-grams to use (e.g., n=3 for trigram model)          |
| model_path | str  | Optional default path for saving and loading the model                 |
| lambdas    | list | Interpolation weights for each n-gram order; defaults to equal weights |

**Functions**

| Method               | Parameters                                    | Returns | Description                                                                                        |
| -------------------- | --------------------------------------------- | ------- | -------------------------------------------------------------------------------------------------- |
| fit                  | training_data: list                           | None    | Fit the n-gram model to a sequence of training tokens.                                             |
| probability          | context: tuple, word: str, alpha: float = 0.4 | float   | Compute the interpolated probability of a word given its context using backoff for unseen n-grams. |
| calculate_perplexity | test_data: list                               | float   | Calculate perplexity of the model on test data.                                                    |
| predict_next_word    | context: list or tuple                        | str     | Predict the most likely next word for a given context.                                             |
| generate_sequence    | length: int = 20, seed: tuple = None          | list    | Generate a sequence of words using the trained model.                                              |
| save                 | filepath: str = None                          | None    | Save the trained n-gram model to disk in a JSON-safe format.                                       |
| load                 | filepath: str = None                          | None    | Load a previously saved n-gram model from disk.                                                    |


**Implementation of Laplace-Smoothing, Backoff and Interpolation**

Since Laplace smoothing, backoff, and interpolation are central concepts in classical n-gram models, we include the relevant code here and provide an explanation of how each concept is implemented.

```python
    def probability(self, context, word, alpha=0.4):
        # Determine which n-gram orders have non-zero interpolation weights
        active_orders = [i + 1 for i, w in enumerate(self.lambdas) if w > 0]
        if not active_orders:
            return 0.0

        prob = 0.0
        for order in sorted(active_orders, reverse=True):
            # Retrieve the interpolation weight for the current n-gram order
            lambda_weight = self.lambdas[order - 1]
            if lambda_weight == 0:  # Skip this order if its interpolation weight is zero (safety check)
                continue

            if order == 1:
                # Apply Laplace smoothing for unigram probability
                prob_order = (self.unigram[word] + 1) / (sum(self.unigram.values()) + self.vocab_size)
            else:
                # Extract the relevant context window for the current n-gram order
                context_slice = tuple(context[-(order - 1):])
                count_context = self.context_counts[order].get(context_slice, 0)
                count_word = self.ngrams[order][context_slice].get(word, 0)
                if count_word > 0:
                    prob_order = count_word / count_context
                else:
                    # Back off to smoothed unigram probability scaled by alpha
                    prob_order = alpha * (self.unigram[word] + 1) / (sum(self.unigram.values()) + self.vocab_size)
            prob += lambda_weight * prob_order

        # Normalize final probability to ensure weights sum correctly
        return prob / sum(self.lambdas)
```
1. *Laplace Smoothing* <br>
   Laplace smoothing is applied in two scenarios within our code:  
   - **Unigram probabilities (order = 1):** When the current n-gram order is `1`, meaning no context is used and we only consider individual word frequencies. This ensures even unseen words get a non-zero probability.  
   - **Backoff:** When the context for a higher-order n-gram is not found, we back off to a smoothed unigram probability scaled by `alpha`.  
        ```python
        (self.unigram[word] + 1) / (sum(self.unigram.values()) + self.vocab_size)
        ```

2. *Interpolation* <br>
   We combine probabilities from multiple n-gram orders (e.g., unigram, bigram, trigram) using predefined interpolation weights stored in `self.lambdas`.  
   This allows the model to balance information from short and long contexts:  
    ```python
    prob += lambda_weight * prob_order
    return prob / sum(self.lambdas)
    ```  
   The normalization by `sum(self.lambdas)` ensures valid probability scaling, even if the provided weights do not sum to one.


3. *Stupid Backoff* <br>
   When a higher-order n-gram `(context + word)` is not found in the training data, we do not recursively reduce the order step by step.  
   Instead, we directly back off to the unigram probability, scaled by a constant factor `alpha`:  
    ```python
    prob_order = alpha * (self.unigram[word] + 1) / (sum(self.unigram.values()) + self.vocab_size)
    ```  
   This simple strategy (known as **stupid backoff**) is computationally efficient and avoids the complexity of full recursive backoff. <br>

**Perplexity Calculation** <br>
Perplexity is calculated by iterating through the test data and summing the log probabilities of each word given its context:  
```python
total_log_prob += math.log(p)
```
where `p` is the predicted probability for the current word.  
The final perplexity is computed as:  
$$
\text{Perplexity} = \exp\left(-\frac{1}{N} \sum_{i=1}^N \log P(w_i|\text{context}) \right)
$$
where \(N\) is the number of words. A lower perplexity value indicates better predictive performance of the model. <br>
<br>
<br>

###### Util Functions
*preprocessing.py - prepare_data* <br>
The prepare_data function initializes the Byte Pair Encoder from above and attempts to load a pre-trained model; if none is found, it trains a new one using the training data and saves it. It then uses this tokenizer to encode the training, validation, and test datasets into tokenized sequences. Finally, it returns the size of the learned BPE vocabulary (merge operations) and the encoded token sequences for each dataset.

| Method        | Parameters                                                      | Returns                                               | Description                                                                                                           |
|---------------|-----------------------------------------------------------------|-------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| prepare_data  | training_data: list, valid_data: list, test_data: list, vocab_size: int = 1000 | tuple (int, list, list, list)                        | Initializes or loads a Byte Pair Encoder model, encodes training, validation, and test datasets, and returns the vocabulary size along with the encoded token sequences for each dataset. |
<br>

###### classical_ngram/main.py


### Neural N-Gram
### GPT

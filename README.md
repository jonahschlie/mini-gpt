# mini-gpt

A step-by-step implementation of a miniature GPT model, built as part of the *Building GPT from Scratch* course. This repository documents the journey of progressively constructing a language model, starting from the fundamentals of text preprocessing and gradually moving towards self-attention and transformer-based architectures.


## Project Overview

The goal of this project is to understand the core principles behind modern language models like GPT by implementing each building block from scratch. This involves:

1. **Text Normalization & Tokenization**  
   - Basic text cleaning and normalization  
   - Character-level tokenization to understand raw sequence modeling  

2. **Byte Pair Encoding (BPE)**  
   - Implemented a subword tokenizer using BPE  
   - Allows a more efficient and expressive tokenization than character-level  

3. **N-gram Language Model**  
   - Built a simple statistical model to predict the next token based on n-grams  
   - Introduced the concept of context windows and probability distributions  

4. **Upcoming Steps**  
   - Self-Attention Mechanisms  
   - Transformer Architectures
   - Pretraining LLMs 

This is an educational and experimental repository intended to deepen understanding of LLM fundamentals.

## Repository Structure
```text
mini-gpt/
├── data/              # shakespeare text data
├── bpe/               # Byte Pair Encoding implementation
├── ngram/             # N-gram language model implementation
├── preprocessing/     # Jupyter notebooks for experiments
├── tests/             # Unit tests
└── README.md          # This file
```

## Getting Started

### Prerequisites
- Python 3.8+
- PyTorch (for future deep learning modules)
- Jupyter Notebook (optional, for exploration)

### Installation
```bash
git clone https://github.com/your-username/mini-gpt.git
cd mini-gpt
pip install -r requirements.txt
```

### Running Examples
Examples will be added as development progresses:
```bash
python ngram/train.py --data data/sample.txt
python bpe/train.py --data data/sample.txt
```

## Roadmap
- [x] Text normalization & character tokenization  
- [x] Byte Pair Encoding (BPE) tokenizer  
- [x] N-gram language model  
- [ ] Self-Attention module  
- [ ] Transformer encoder/decoder blocks  
- [ ] Mini GPT training pipeline  

## Contributing
This repository is primarily for personal learning, but suggestions and improvements are welcome.

## License
This project is licensed under the MIT License.

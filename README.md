# Machine-Translation
Machine Translation using Neural Networks
Neural Machine Translation Research Project
Comparative Study of Sequence-to-Sequence Architectures for Low-Resource Neural Machine Translation

This project implements and analyzes multiple Neural Machine Translation (NMT) architectures for translating English → French. The goal is to study the effectiveness of different sequence-to-sequence models and modern transformer-based architectures in a controlled experimental environment.

The project progresses from simple recurrent models to advanced architectures and evaluates them using standard machine translation metrics.

Project Overview

Machine Translation is one of the most challenging problems in Natural Language Processing (NLP). It involves automatically translating text from one language to another while preserving meaning, grammar, and context.

Traditional statistical methods have been replaced by Neural Machine Translation (NMT) models that learn representations directly from data.

This project explores and compares:

Basic Recurrent Neural Networks (RNN)

RNN with Embedding

Attention-based Seq2Seq models

Transformer architecture implemented from scratch

The project also integrates advanced training techniques such as:

Beam Search Decoding

Label Smoothing

Scheduled Sampling

Mixed Precision Training

Ablation Studies

BLEU Score Evaluation

Research Motivation

Most introductory NMT implementations demonstrate only one model. However, real-world machine translation systems require rigorous comparison between architectures.

This project addresses the following research questions:

How do different sequence models perform on low-resource translation tasks?

Does the transformer architecture outperform recurrent models?

How do embedding size and attention heads affect performance?

What decoding strategies improve translation quality?

The project evaluates these questions through systematic experimentation.

Dataset

The dataset consists of parallel English–French sentence pairs.

Example:

English

new jersey is sometimes quiet during autumn , and it is snowy in april .

French

new jersey est parfois calme pendant l' automne , et il est neigeux en avril .

Each line in the dataset corresponds to a translation pair.

Data Preprocessing

Before training the models, the dataset undergoes several preprocessing steps.

Tokenization

Text is converted into sequences of integer tokens using a tokenizer.

Example:

Input sentence

The quick brown fox jumps

Tokenized output

[1, 2, 4, 5, 6]
Padding

Sequences are padded to ensure equal length across the dataset.

Example

[1 2 4 5 6]
[3 7 9]

After padding

[1 2 4 5 6]
[3 7 9 0 0]
Vocabulary Construction

Vocabulary statistics from the dataset:

Language	Total Words	Unique Words
English	~1.8M	~227
French	~1.9M	~355
Model Architectures

The project implements several machine translation architectures.

1. Simple RNN Model

The baseline model uses a GRU-based recurrent neural network.

Architecture

Input → GRU → Dense → Softmax

Limitations

Difficulty capturing long-range dependencies

Lower translation quality

Baseline accuracy achieved

~60%
2. RNN with Word Embeddings

This model adds a trainable embedding layer.

Architecture

Input → Embedding → GRU → Dense → Softmax

Benefits

Captures semantic relationships between words

Faster convergence

Improved translation quality

Accuracy achieved

~84%
3. Attention-based Seq2Seq Model

Attention allows the decoder to focus on specific words in the source sentence.

Benefits

Better alignment between source and target sentences

Improved handling of long sentences

4. Transformer Model (Implemented From Scratch)

The transformer architecture removes recurrence entirely and relies on attention mechanisms.

Key components implemented manually:

Scaled Dot Product Attention
Attention(Q,K,V) = softmax(QKᵀ / √dk)V
Multi-Head Attention

Multiple attention heads learn different contextual relationships.

Positional Encoding

Since transformers lack recurrence, positional encoding provides sequence order information.

Masked Decoder Attention

Prevents the model from accessing future tokens during training.

Transformer architecture

Input → Embedding → Positional Encoding
       → Multi-Head Attention
       → Feed Forward Network
       → Decoder
       → Softmax
Training Techniques

Several techniques were used to improve model performance.

Beam Search Decoding

Instead of greedy decoding, beam search keeps the best candidate sequences.

Benefits

Higher translation quality

Better sentence structure

Label Smoothing

Prevents the model from becoming overconfident.

True label probability = 0.9
Remaining probability distributed across other classes
Scheduled Sampling

Gradually replaces ground truth tokens with predicted tokens during training.

Benefits

Reduces exposure bias

Improves robustness

Mixed Precision Training

Uses float16 operations for faster GPU training.

Benefits

Faster training

Lower memory usage

Evaluation Metrics

The project evaluates models using standard machine translation metrics.

BLEU Score

BLEU measures similarity between predicted translations and reference translations.

Higher BLEU scores indicate better translations.

Example

Model	BLEU
RNN	21.4
Attention	27.9
Transformer	32.5
Perplexity

Measures how well the model predicts the next word.

Lower perplexity indicates better language modeling.

Token Accuracy

Measures the percentage of correctly predicted tokens.

Ablation Study

Experiments were conducted to analyze the effect of different architectural components.

Experiment	BLEU
Transformer (baseline)	32.5
Without positional encoding	26.1
2 attention heads	29.4
4 attention heads	31.7
Embedding size 64	28.9
Embedding size 256	33.2

These experiments demonstrate the importance of attention heads and embedding size in translation quality.

Project Structure
nmt_research_project
│
├── configs
│   └── config.py
│
├── data
│   └── dataset_loader.py
│
├── preprocessing
│   ├── tokenizer.py
│   └── padding.py
│
├── models
│   ├── rnn_model.py
│   ├── attention_model.py
│   └── transformer_from_scratch.py
│
├── training
│   └── trainer.py
│
├── decoding
│   └── beam_search.py
│
├── evaluation
│   ├── bleu_score.py
│   ├── perplexity.py
│   └── token_accuracy.py
│
├── experiments
│   └── ablation_study.py
│
├── utils
│   └── model_io.py
│
└── main.py
Running the Project
Install Dependencies
pip install tensorflow keras numpy nltk
Train Model
python main.py --model transformer

Options

--model rnn
--model attention
--model transformer
Run Evaluation
python evaluate.py
Run Ablation Experiments
python experiments/ablation_study.py
Example Translation

Input

the weather is cold today

Predicted Translation

le temps est froid aujourd'hui
Future Improvements

Possible extensions include:

Multilingual translation

Pretrained transformer fine-tuning

Larger datasets such as WMT

Subword tokenization (BPE)

Reinforcement learning for sequence generation

Applications

Neural Machine Translation has applications in

Cross-language communication

International content localization

Multilingual search systems

Real-time translation tools

Research Relevance

This project demonstrates key machine learning concepts:

Sequence modeling

Attention mechanisms

Transformer architectures

Experimental research methodology

The repository is structured to resemble real-world machine learning research projects developed in industry and academia.

Author

Ayush Raj
B.Tech Computer Science (Data Science)

Interests

Artificial Intelligence

Natural Language Processing

Large Scale Machine Learning Systems

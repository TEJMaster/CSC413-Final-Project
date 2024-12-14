# Movie Review Categorization with RNN and Transformer Baselines

This repository contains implementations of two baseline models for sentiment analysis on the IMDB movie reviews dataset:

1. **RNN Baseline**: A Recurrent Neural Network (RNN) with Long Short-Term Memory (LSTM) layers.
2. **Transformer Baseline**: A model leveraging self-attention mechanisms.

These baselines establish benchmarks for future experiments with more advanced models.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
  - [Dataset Repository](#dataset-repository)
  - [Dataset Preparation](#dataset-preparation)
- [Baseline Models](#baseline-models)
  - [RNN Baseline](#rnn-baseline)
  - [Transformer Baseline](#transformer-baseline)
- [Training Process](#training-process)
- [Evaluation](#evaluation)
- [Potential Improvements](#potential-improvements)
- [Usage](#usage)
  - [RNN Baseline Usage](#rnn-baseline-usage)
  - [Transformer Baseline Usage](#transformer-baseline-usage)
- [Additional Resources](#additional-resources)
- [License](#license)

## Introduction

Sentiment analysis is a key task in NLP that involves classifying the sentiment expressed in text. This project uses the IMDB movie reviews dataset to develop two baseline models. The goal is to classify movie reviews as either positive or negative.

## Dataset

### Dataset Repository

The dataset used in this project is sourced from the [IMDB Movie Reviews Dataset](https://ai.stanford.edu/~amaas/data/sentiment/).

### Dataset Preparation

**Steps Taken**:
1. The dataset, originally in text format, was converted into CSV files for easier handling.
2. Files are organized as follows:
   - `train_neg.csv`: Negative reviews for training.
   - `train_pos.csv`: Positive reviews for training.
   - `test_neg.csv`: Negative reviews for testing.
   - `test_pos.csv`: Positive reviews for testing.
3. Data was tokenized and numericalized into fixed-length sequences.

## Baseline Models

### RNN Baseline

The RNN baseline model uses Long Short-Term Memory (LSTM) layers. It consists of:
- **Embedding Layer**: Transforms input tokens into dense vector representations.
- **LSTM Layers**: Captures temporal dependencies in the sequence.
- **Fully Connected Layer**: Maps the LSTM outputs to a binary classification task.

### Transformer Baseline

The Transformer baseline leverages self-attention mechanisms for text representation. The key components include:
- **Embedding Layer**: Maps input tokens to dense vector embeddings.
- **Positional Encoding**: Adds sequence position information to embeddings.
- **Transformer Encoder Layers**: Captures contextual information through multi-head attention and feedforward networks.

## Training Process

Both models were trained using the same pipeline:
1. Preprocessed data into tokenized and numericalized sequences.
2. Split data into training, validation, and test sets.
3. Used PyTorch DataLoaders for efficient batching.
4. Optimized using Adam with a learning rate scheduler.
5. Saved the best-performing model based on validation loss.

## Evaluation

Model performance was evaluated using accuracy and confusion matrices. Additionally, visualizations of training and validation loss/accuracy were plotted over epochs.

## Potential Improvements

- **Pre-trained Embeddings**: Using embeddings like GloVe or Word2Vec.
- **Attention in RNNs**: Adding attention mechanisms to the RNN model.
- **Hyperparameter Tuning**: Systematic optimization of learning rate, batch size, etc.
- **Augmenting Transformers**: Exploring variations such as BERT or GPT architectures.

## Usage

### RNN Baseline Usage

1. **Run the RNN model**:
   ```bash
   python rnn_baseline.py
   ```

### Transformer Baseline Usage

1. **Run the Transformer model**:
   ```bash
   python transformer_baseline.py
   ```

## Additional Resources

- **IMDB Movie Reviews Dataset**: The original dataset used in this project for sentiment analysis. [IMDB Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)

- **Sentiment Analysis using RNN-LSTM**: This repository demonstrates sentiment analysis using Recurrent Neural Networks (RNN) with Long Short-Term Memory (LSTM) units, employing the Google News Word2Vec model for word embeddings. [GitHub Repository](https://github.com/saadarshad102/Sentiment-Analysis-RNN-LSTM)

- **Sentiment Analysis with Transformers**: This project implements sentiment analysis utilizing the BERT model from the Transformers library, providing a step-by-step guide on data preprocessing, model training, evaluation, and sample predictions. [GitHub Repository](https://github.com/Guacamoley/Transformers-Sentiment-Analysis)

## License

This project is licensed under the MIT License. See the LICENSE file for details.

# NLP Baselines Project

This repository encompasses two core Natural Language Processing (NLP) tasks, each implementing baseline models using Recurrent Neural Networks (RNN) and Transformer architectures. These tasks serve as foundational benchmarks for future experiments and model enhancements.

## Table of Contents

- [NLP Baselines Project](#nlp-baselines-project)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Task 1: Movie Review Categorization](#task-1-movie-review-categorization)
    - [Dataset](#dataset)
    - [Models](#models)
    - [Training \& Evaluation](#training--evaluation)
    - [Data Augmentation](#data-augmentation)
  - [Task 2: English-French Translation](#task-2-english-french-translation)
    - [Dataset](#dataset-1)
    - [Models](#models-1)
    - [Training \& Evaluation](#training--evaluation-1)
    - [Data Augmentation](#data-augmentation-1)
  - [Additional Resources](#additional-resources)
  - [License](#license)

## Introduction

This project aims to establish baseline models for two pivotal NLP tasks:

1. **Movie Review Categorization**: Classifying IMDB movie reviews as positive or negative.
2. **English-French Translation**: Translating English sentences into French.

Both tasks utilize RNN-based models and Transformer architectures to provide a comparative analysis of traditional and state-of-the-art approaches.

## Task 1: Movie Review Categorization

### Dataset

- **Source**: [IMDB Movie Reviews Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
- **Preparation**:
  - Converted text data into CSV format:
    - `train_neg.csv`, `train_pos.csv`: Training data.
    - `test_neg.csv`, `test_pos.csv`: Testing data.
  - Tokenized and numericalized reviews into fixed-length sequences.

### Models

1. **RNN Baseline**:
   - **Architecture**:
     - **Embedding Layer**: Utilizes pre-trained [GloVe embeddings](https://nlp.stanford.edu/projects/glove/) to transform input tokens into dense vector representations.
     - **LSTM Layers**
     - **Fully Connected Layer** for binary classification
2. **Transformer Baseline**:
   - **Architecture**:
     - **Embedding Layer with Positional Encoding**: Incorporates pre-trained GloVe embeddings for input token representation.
     - **Transformer Encoder Layers**
     - **Classification Head**
   - **Note**: Custom-built using PyTorch’s `nn.TransformerEncoder`, without pre-trained models like BERT.

### Training & Evaluation

- **Process**:
  - Data split into training, validation, and test sets.
  - Employed data augmentation for robustness.
  - Utilized PyTorch DataLoaders for batching.
  - Trained models and saved the best-performing based on validation loss.
- **Metrics**:
  - Accuracy
  - Confusion Matrices
  - Training/Validation Loss plots
  - BLEU Scores (for Transformer)

### Augmentation

- **Strategies**:
  - Varying proportions of positive samples.
  - Synonym-based augmentations.
  - Adjusting training data size.
  - Incorporating pre-trained embeddings.
- **Implementation**: Organized Jupyter notebooks reflecting each augmentation strategy for reproducibility.

## Task 2: English-French Translation

### Dataset

- **Source**: [Laurent Veyssier's Machine Translation Repository](https://github.com/LaurentVeyssier/Machine-translation-English-French-with-Deep-neural-Network/tree/main/data)
- **Preparation**:
  - Data Cleaning for consistency.
  - Splitting into training, validation, and test sets.
  - Building vocabularies for English and French.

### Models

1. **RNN Baseline (GRU-based)**:
   - **Architecture**:
     - **Encoder**:
       - **Embedding Layer**: Utilizes pre-trained [GloVe embeddings](https://nlp.stanford.edu/projects/glove/) for English token representation.
       - **GRU Layers**
     - **Decoder**:
       - **Embedding Layer**: Utilizes pre-trained GloVe embeddings for French token representation.
       - **GRU Layers**
       - **Fully Connected Layer**
     - **Seq2Seq Wrapper** for end-to-end training
2. **Transformer Baseline**:
   - **Architecture**:
     - **Encoder and Decoder**:
       - **Embedding Layer with Positional Encoding**: Incorporates pre-trained GloVe embeddings.
       - **Multi-Head Self-Attention and Feed-Forward Layers**
     - **Seq2Seq Wrapper** for end-to-end training
   - **Note**: Custom-built using PyTorch’s `nn.TransformerEncoder`, without pre-trained models.

### Training & Evaluation

- **Process**:
  - Data tokenization and vocabulary mapping.
  - Applied teacher forcing during training.
  - Employed data augmentation for model robustness.
  - Saved the best-performing models based on validation loss.
- **Metrics**:
  - Cross-Entropy Loss
  - BLEU Scores
  - Training/Validation Loss and BLEU Score visualizations

### Augmentation

- **Strategies**:
  - Varying the proportion of long sentences.
  - Adjusting training data size.
  - Incorporating pre-trained embeddings.
  - Modifying the number of attention heads in the Transformer.
- **Implementation**: Organized experiments with corresponding visualizations for each augmentation strategy.

## Additional Resources

- **Sentiment Analysis using RNN-LSTM**: [GitHub Repository](https://github.com/saadarshad102/Sentiment-Analysis-RNN-LSTM)
- **Transformer References**:
  - [NLP From Scratch: Seq2Seq with Attention](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)
  - [Neural Machine Translation with a Transformer and Keras](https://www.tensorflow.org/text/tutorials/transformer)
  - [Neural Machine Translation with Attention in TensorFlow](https://www.tensorflow.org/text/tutorials/nmt_with_attention)
- **BLEU Score Implementation**: [BangoC123/BLEU](https://github.com/bangoc123/BLEU)
- **GloVe Embeddings**: [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)
  **Description**: Pre-trained word vectors developed by Stanford University, widely used for embedding layers in NLP models.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

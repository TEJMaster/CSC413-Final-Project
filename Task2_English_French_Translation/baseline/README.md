# English-French Translation with RNN and Transformer Baselines

This project focuses on building and benchmarking machine translation models to translate English sentences into French. Two distinct architectures are provided as benchmarks:

1. **RNN Baseline (GRU-based)**: A traditional approach leveraging Recurrent Neural Networks with Gated Recurrent Units (GRUs) to model sequential data.
2. **Transformer Baseline**: A modern approach utilizing Transformer architectures with self-attention mechanisms to capture global dependencies within text.

By implementing and evaluating these two models, we aim to understand the performance improvements and trade-offs between classic sequence models and state-of-the-art Transformer architectures in the context of machine translation.

## Table of Contents

- [English-French Translation with RNN and Transformer Baselines](#english-french-translation-with-rnn-and-transformer-baselines)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Dataset](#dataset)
    - [Dataset Repository](#dataset-repository)
    - [Dataset Preparation](#dataset-preparation)
  - [Models](#models)
    - [RNN Baseline (GRU-based)](#rnn-baseline-gru-based)
    - [Transformer Baseline](#transformer-baseline)
  - [Training Process](#training-process)
  - [Evaluation](#evaluation)
  - [Additional Resources](#additional-resources)
  - [License](#license)

## Introduction

Machine Translation (MT) is a pivotal task in Natural Language Processing (NLP) that involves automatically translating text from one language to another. This project utilizes the English-French language pair to build and benchmark two different MT models:

- **RNN (GRU) Baseline**: Utilizes Recurrent Neural Networks with GRU layers to capture sequential dependencies in language data.
- **Transformer Baseline**: Employs Transformer architectures with self-attention mechanisms to model long-range dependencies and contextual relationships within text.

Establishing these baselines is crucial for comparing the effectiveness of different model architectures and guiding future enhancements and optimizations in machine translation systems.

## Dataset

### Dataset Repository

The dataset used for this project is sourced from the [Laurent Veyssier's Machine Translation Repository](https://github.com/LaurentVeyssier/Machine-translation-English-French-with-Deep-neural-Network/tree/main/data). It consists of parallel English-French sentence pairs, which are essential for training and evaluating machine translation models.

### Dataset Preparation

**Steps Taken:**

1. **Data Acquisition**: Retrieved the dataset from the official [GitHub repository](https://github.com/LaurentVeyssier/Machine-translation-English-French-with-Deep-neural-Network/tree/main/data).
2. **Data Cleaning**: Ensured that the dataset is free from irregularities such as inconsistent formatting or missing values.
3. **Data Splitting**: The dataset is divided into training, validation, and test sets to facilitate model training and unbiased evaluation.
4. **Vocabulary Building**: Constructed vocabularies for both English and French languages, mapping each unique token to an integer index.

## Models

### RNN Baseline (GRU-based)

The RNN baseline employs a Sequence-to-Sequence (Seq2Seq) architecture with GRU layers for both the encoder and decoder. This model is designed to translate English sentences into French by learning the sequential dependencies in the data.

**Architecture Components:**

1. **Encoder (GRU-based)**:
   - **Embedding Layer**: Transforms input English tokens into dense vector representations.
   - **GRU Layers**: Capture the sequential dependencies in the input data.
   
2. **Decoder (GRU-based)**:
   - **Embedding Layer**: Transforms input French tokens (shifted right) into dense vectors.
   - **GRU Layers**: Generate the output sequence based on the encoder's context.
   - **Fully Connected Layer**: Maps GRU outputs to the French vocabulary space for prediction.

3. **Seq2Seq Wrapper**:
   - Combines the encoder and decoder to facilitate end-to-end training.

### Transformer Baseline

The Transformer baseline utilizes a Transformer-based Sequence-to-Sequence architecture, which has demonstrated superior performance in various NLP tasks due to its ability to model long-range dependencies through self-attention mechanisms.

**Architecture Components:**

1. **Encoder**:
   - **Embedding Layer**: Converts input English tokens into dense vector representations.
   - **Positional Encoding**: Adds positional information to the embeddings to retain the order of tokens.
   - **Transformer Encoder Layers**: Consist of multi-head self-attention and feed-forward neural networks to process the embeddings.

2. **Decoder**:
   - **Embedding Layer**: Converts input French tokens (shifted right) into dense vectors.
   - **Positional Encoding**: Adds positional information to the embeddings.
   - **Transformer Decoder Layers**: Includes masked multi-head self-attention and encoder-decoder attention mechanisms.
   - **Fully Connected Layer**: Maps Transformer outputs to the French vocabulary space for prediction.

3. **Seq2Seq Wrapper**:
   - Combines the encoder and decoder to facilitate end-to-end training.

**Note**: This Transformer implementation is custom-built using PyTorch's `nn.TransformerEncoder` and does **not** utilize pre-trained Transformer architectures like BERT.

## Training Process

The training workflow includes:

1. **Data Preparation**: Tokenization, vocabulary building, and splitting the dataset into training, validation, and test sets.
2. **Data Augmentation**: Employed various data augmentation techniques to enhance model robustness, as detailed in the [Data Augmentation Tests](#data-augmentation-tests) section.
3. **Model Training**: Both RNN and Transformer models are trained on the prepared dataset using teacher forcing.
4. **Evaluation**: Evaluate models on the validation and test datasets, calculating loss and BLEU scores.
5. **Model Saving**: Saved the best-performing model based on validation loss for future use and inference.

## Evaluation

The evaluation metrics include:

1. **Cross-Entropy Loss**: Measures the difference between predicted and actual tokens.
2. **BLEU Score**: Evaluates the quality of translations by comparing predicted translations to the reference translations.

Model performance was evaluated using accuracy and confusion matrices. Additionally, visualizations of training and validation loss/BLEU scores were plotted over epochs to monitor the training progress and assess model convergence.

**Descriptions of Augmentation Tests:**

- **Proportion-Based Augmentations:**
  - `rnn_20%long.png` to `rnn_80%long.png`: Experiments varying the percentage of long sentences (sentence length > 15) in the RNN training data.
  - `transformer_20%long.png` to `transformer_80%long.png`: Similar experiments for the Transformer baseline.

- **Training Data Size Adjustments:**
  - `rnn_15000data.png` & `rnn_5000data.png`: Testing the impact of different amounts of training data for the RNN model.
  - `transformer_15000data.png` & `transformer_5000data.png`: Similar adjustments for the Transformer model.

- **Baseline Results:**
  - `rnn_baseline.png` & `transformer_baseline_generative.txt`: Added baseline results for both models.
  - `rnn_baseline_generative.txt` & `transformer_baseline_generative.txt`: Additional results related to generative tasks.

- **Pre-trained Embedding Augmentations:**
  - `rnn_pre_trained_english.png` & `transformer_pre_trained_english.png`: Incorporating pre-trained embeddings to enhance model performance.

- **Transformer Specific Augmentations:**
  - `transformer_baseline_4head.png`, `transformer_baseline_8head.png`, `transformer_baseline_16head.png`: Testing the effect of varying the number of attention heads in the Transformer model.

## Additional Resources

- **RNN Baseline Reference**: [NLP From Scratch: Translation with a Sequence to Sequence Network and Attention](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)
- **Transformer Baseline Reference**: [Neural Machine Translation with a Transformer and Keras](https://www.tensorflow.org/text/tutorials/transformer)
- **NMT with Attention**: [Neural Machine Translation with Attention in TensorFlow](https://www.tensorflow.org/text/tutorials/nmt_with_attention)
- **Data Source**: The English-French parallel corpus used in this project is sourced from [Laurent Veyssier's Machine Translation Repository](https://github.com/LaurentVeyssier/Machine-translation-English-French-with-Deep-neural-Network/tree/main/data).
- **Evaluation Metric**: BLEU score computation is based on the implementation from [BangoC123/BLEU](https://github.com/bangoc123/BLEU).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

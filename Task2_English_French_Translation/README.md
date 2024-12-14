# English-French Translation

This project focuses on building and benchmarking machine translation models to translate English sentences into French. Two distinct architectures are provided as benchmarks:

1. **RNN Baseline (GRU-based)**: A traditional approach leveraging Recurrent Neural Networks with Gated Recurrent Units (GRUs) to model sequential data.
2. **Transformer Baseline**: A modern approach utilizing Transformer architectures with self-attention mechanisms to capture global dependencies within text.

By implementing and evaluating these two models, we aim to understand the performance improvements and trade-offs between classic sequence models and state-of-the-art Transformer architectures in the context of machine translation.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
  - [Dataset Preparation](#dataset-preparation)
  - [Dataset Files](#dataset-files)
- [Models](#models)
  - [RNN Baseline (GRU-based)](#rnn-baseline-gru-based)
  - [Transformer Baseline](#transformer-baseline)
- [Training Process](#training-process)
  - [Common Steps](#common-steps)
  - [RNN Training Steps](#rnn-training-steps)
  - [Transformer Training Steps](#transformer-training-steps)
- [Evaluation](#evaluation)
- [Potential Improvements](#potential-improvements)
- [Dataset Repository](#dataset_repository)
- [Usage](#usage)
  - [RNN Baseline Usage](#rnn-baseline-usage)
  - [Transformer Baseline Usage](#transformer-baseline-usage)
- [License](#license)

## Introduction

Machine Translation (MT) is a pivotal task in Natural Language Processing (NLP) that involves automatically translating text from one language to another. This project utilizes the English-French language pair to build and benchmark two different MT models:

- **RNN (GRU) Baseline**: Utilizes Recurrent Neural Networks with GRU layers to capture sequential dependencies in language data.
- **Transformer Baseline**: Employs Transformer architectures with self-attention mechanisms to model long-range dependencies and contextual relationships within text.

Establishing these baselines is crucial for comparing the effectiveness of different model architectures and guiding future enhancements and optimizations in machine translation systems.

## Dataset

### Dataset Preparation

The dataset used for this project is sourced from the [Laurent Veyssier's Machine Translation Repository](https://github.com/LaurentVeyssier/Machine-translation-English-French-with-Deep-neural-Network/tree/main/data). It consists of parallel English-French sentence pairs, which are essential for training and evaluating machine translation models.

**Steps Taken:**

1. **Data Acquisition**: Retrieved the dataset from the official [GitHub repository](https://github.com/LaurentVeyssier/Machine-translation-English-French-with-Deep-neural-Network/tree/main/data).
2. **Data Cleaning**: Ensured that the dataset is free from irregularities such as inconsistent formatting or missing values.
3. **Data Splitting**: The dataset is divided into training, validation, and test sets to facilitate model training and unbiased evaluation.
4. **Vocabulary Building**: Constructed vocabularies for both English and French languages, mapping each unique token to an integer index.

### Dataset Files

All dataset files are located in the `translation_train` directory within this repository. The key files include:

- **`small_vocab_en.txt`**: Contains English sentences for training and evaluation.
- **`small_vocab_fr.txt`**: Contains corresponding French translations for the English sentences.

**Note:** The dataset has been limited to a manageable size (`sample_max = 10,000` samples) to expedite training and experimentation. Adjust `sample_max` in the scripts if you wish to use a larger dataset.

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

## Training Process

### Common Steps

Both models follow a similar training workflow with shared preprocessing and data handling steps:

1. **Environment Setup**: Install necessary libraries and set up the environment for PyTorch and NLP processing.
2. **Data Loading**: Load the English-French sentence pairs from the CSV files.
3. **Tokenization**: Use `nltk` for efficient tokenization to convert text into tokens.
4. **Vocabulary Building**: Construct vocabularies for English and French languages, mapping each unique token to an integer index.
5. **Numericalization**: Convert tokenized sentences into numerical indices based on the constructed vocabularies.
6. **Dataset and DataLoader Creation**: Implement custom PyTorch `Dataset` and `DataLoader` classes to handle batching and padding of sequences.
7. **Model Initialization**: Initialize the respective model architectures with specified hyperparameters.
8. **Training**: Train the models for a set number of epochs, monitoring validation loss to save the best-performing model.
9. **Evaluation**: Evaluate the best model on the test set and record performance metrics.
10. **Visualization**: Visualize training and validation loss and BLEU scores over epochs.
11. **Prediction**: Use the trained models to translate custom English sentences into French.

### RNN Training Steps

The RNN training process involves:

1. **Model Definition**: Define the Seq2Seq architecture with GRU-based encoder and decoder.
2. **Loss and Optimizer**: Use Cross-Entropy Loss with padding index ignored and the Adam optimizer.
3. **Training Loop**: Iterate over epochs, training the model on the training set and evaluating on the validation set.
4. **Model Saving**: Save the best-performing model based on validation loss.
5. **BLEU Score Computation**: Calculate the BLEU score on the validation set to assess translation quality.

### Transformer Training Steps

The Transformer training process includes:

1. **Model Definition**: Define the Transformer-based Seq2Seq architecture with encoder and decoder layers.
2. **Loss and Optimizer**: Utilize Cross-Entropy Loss with padding index ignored and the Adam optimizer.
3. **Training Loop**: Iterate over epochs, training the model on the training set and evaluating on the validation set.
4. **Model Saving**: Save the best-performing model based on validation loss.
5. **BLEU Score Computation**: Calculate the BLEU score on the validation set to assess translation quality.

## Evaluation

Both models are evaluated using the following metrics:

- **Loss**: Cross-Entropy Loss to measure the discrepancy between predicted and actual tokens.
- **BLEU Score**: A standard metric for evaluating the quality of machine-translated text by comparing it to one or more reference translations.

**Evaluation Process:**

1. **Validation Evaluation**: During training, after each epoch, evaluate the model on the validation set to monitor performance and prevent overfitting.
2. **Test Evaluation**: After training, evaluate the best model on the test set to assess its generalization capability.
3. **BLEU Score Calculation**: Compute the BLEU score on a subset of the validation and test sets to quantitatively measure translation quality.
4. **Visualization**: Plot training and validation loss alongside BLEU scores to visualize the model's learning progress.

## Potential Improvements

While both baseline models provide solid foundations, several areas offer opportunities for enhancement:

- **Pre-trained Embeddings**: Incorporating pre-trained word embeddings like GloVe or FastText could enhance the models' understanding of semantic relationships.
- **Attention Mechanisms**: Experimenting with more sophisticated attention mechanisms or increasing the number of attention heads in the Transformer model might yield better performance.
- **Hyperparameter Optimization**: Systematically exploring hyperparameters (e.g., learning rate, batch size, embedding size) could lead to improved model performance.
- **Beam Search Decoding**: Implementing beam search instead of greedy decoding can improve translation quality by considering multiple possible translations.
- **Handling Larger Datasets**: Scaling up the dataset size could enhance the models' ability to generalize and improve translation accuracy.
- **Incorporating Bidirectional RNNs**: Using bidirectional GRUs in the encoder could provide the decoder with more contextual information.

## Dataset Repository
- **`small_vocab_en.txt`**: English sentences.
- **`small_vocab_fr.txt`**: Corresponding French translations.
- **`rnn_baseline_generative.py`**: The main Python script implementing the RNN (GRU-based) baseline translation model.
- **`transformer_baseline_generative.py`**: The main Python script implementing the Transformer-based baseline translation model.
- **`README.md`**: This documentation file.
## Usage

### RNN Baseline Usage

To replicate the RNN (GRU-based) baseline model for English-French translation:

1. **Clone the Repository**

   ```bash
   git clone https://github.com/YourUsername/English-French-Translation.git
   cd English-French-Translation
   ```

2. **Install Dependencies**

   It's recommended to use a virtual environment.

   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the Dataset**

   Ensure that the `translation_train` directory contains the necessary data files (`small_vocab_en.txt`, `small_vocab_fr.txt`).

4. **Run the RNN Baseline Model**

   Execute the Python script to train and evaluate the RNN model.

   ```bash
   python rnn_baseline_generative.py
   ```

   **What Happens:**

   - The model is trained for the specified number of epochs (`num_epochs = 40`).
   - After each epoch, the model is evaluated on the validation set, and the BLEU score is computed.
   - The best-performing model based on validation loss is saved as `best-model.pt`.
   - After training, the model is evaluated on the test set, and a BLEU score is reported.
   - Visualizations of training/validation loss and BLEU scores are displayed.
   - The script demonstrates translation of a random test sentence and a custom English sentence.

### Transformer Baseline Usage

To replicate the Transformer baseline model for English-French translation:

1. **Clone the Repository**

   ```bash
   git clone https://github.com/YourUsername/English-French-Translation.git
   cd English-French-Translation
   ```

2. **Install Dependencies**

   It's recommended to use a virtual environment.

   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the Dataset**

   Ensure that the `translation_train` directory contains the necessary data files (`small_vocab_en.txt`, `small_vocab_fr.txt`).

4. **Run the Transformer Baseline Model**

   Execute the Python script to train and evaluate the Transformer model.

   ```bash
   python transformer_baseline_generative.py
   ```

   **What Happens:**

   - The model is trained for the specified number of epochs (`num_epochs = 40`).
   - After each epoch, the model is evaluated on the validation set, and the BLEU score is computed.
   - The best-performing model based on validation loss is saved as `best-transformer-model.pt`.
   - After training, the model is evaluated on the test set, and a BLEU score is reported.
   - Visualizations of training/validation loss and BLEU scores are displayed.
   - The script demonstrates translation of a random test sentence and a custom English sentence.

## License

This project is licensed under the [MIT License](LICENSE).

---

**Acknowledgments:**

- **Data Source**: The English-French parallel corpus used in this project is sourced from [Laurent Veyssier's Machine Translation Repository](https://github.com/LaurentVeyssier/Machine-translation-English-French-with-Deep-neural-Network/tree/main/data).
- **Evaluation Metric**: BLEU score computation is based on the implementation from [BangoC123/BLEU](https://github.com/bangoc123/BLEU).

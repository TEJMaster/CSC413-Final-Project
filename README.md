# Sentiment Analysis with RNN Baseline

This repository contains the implementation of a baseline Recurrent Neural Network (RNN) model using Long Short-Term Memory (LSTM) layers for sentiment analysis on the IMDB movie reviews dataset. The primary objective of this project is to establish a performance benchmark using RNNs, which will later be compared against more advanced Transformer-based models with attention mechanisms.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
  - [Dataset Preparation](#dataset-preparation)
  - [Dataset Files](#dataset-files)
- [Baseline RNN Model](#baseline-rnn-model)
  - [Model Architecture](#model-architecture)
  - [Training Process](#training-process)
- [Potential Improvements](#potential-improvements)
- [Training Steps](#training-steps)
- [Conclusion](#conclusion)
- [Repository Structure](#repository-structure)
- [Usage](#usage)
- [License](#license)

## Introduction

Sentiment analysis is a fundamental task in Natural Language Processing (NLP) that involves determining the sentiment expressed in a piece of text. This project focuses on building a baseline RNN model with LSTM layers to classify IMDB movie reviews as positive or negative. Establishing this baseline is crucial for comparing the performance of more sophisticated models, such as Transformers with attention mechanisms.

## Dataset

### Dataset Preparation

The dataset used for this project is derived from the [IMDB Movie Reviews](https://ai.stanford.edu/~amaas/data/sentiment/) dataset, which contains a large number of movie reviews labeled as positive or negative. To facilitate easier handling and processing within the model, the original text files were manually converted into CSV format. This conversion was done to streamline data loading and preprocessing steps in the training pipeline.

### Dataset Files

All dataset files are located in the `rnn_baseline_dataset` directory within this repository. The key files include:

- **`train_neg.csv`**: Contains negative movie reviews for training.
- **`train_pos.csv`**: Contains positive movie reviews for training.
- **`test_neg.csv`**: Contains negative movie reviews for testing.
- **`test_pos.csv`**: Contains positive movie reviews for testing.
- **`train_unsup.csv`**: Contains unlabeled movie reviews for unsupervised training (not used in this baseline).

**Note:** The `train_unsup.csv` file was included in the dataset but was not utilized in this baseline implementation. The primary reason for not using this file is to maintain a supervised learning setup, focusing solely on labeled data to establish a clear performance benchmark before introducing unsupervised or semi-supervised techniques.

## Baseline RNN Model

### Model Architecture

The baseline model employs a Recurrent Neural Network with LSTM layers, which are well-suited for capturing sequential dependencies in text data. The architecture comprises the following components:

1. **Embedding Layer**: Transforms input tokens into dense vector representations.
2. **LSTM Layers**: Capture the temporal dependencies in the sequence of embeddings.
3. **Fully Connected Layer**: Maps the LSTM outputs to the desired output dimension (binary classification).
4. **Dropout Layer**: Regularizes the model to prevent overfitting.

### Training Process

The model was trained using the following process:

1. **Data Loading**: CSV files were loaded and split into training, validation, and test sets.
2. **Tokenization**: Reviews were tokenized using `spaCy` to convert text into tokens.
3. **Vocabulary Building**: A vocabulary was constructed from the training data, mapping each unique token to an integer index.
4. **Dataset and DataLoader Creation**: Custom PyTorch `Dataset` and `DataLoader` classes were implemented to handle batching and padding of sequences.
5. **Model Initialization**: The RNN model was initialized with specified hyperparameters, including embedding dimension, hidden dimension, number of layers, and dropout rate.
6. **Training**: The model was trained for a set number of epochs, monitoring validation loss to save the best-performing model.
7. **Evaluation**: The best model was evaluated on the test set, and performance metrics were recorded.
8. **Visualization**: Training and validation loss and accuracy were visualized over epochs.
9. **Prediction**: The trained model was used to predict sentiments of custom sentences.

## Potential Improvements

While the RNN baseline provides a solid foundation, there are several areas where the model can be enhanced:

- **Pre-trained Embeddings**: Incorporating pre-trained word embeddings like GloVe or Word2Vec could improve the model's ability to understand semantic relationships.
- **Bidirectional LSTM**: Although a bidirectional LSTM was used, further tuning its parameters or experimenting with different architectures could yield better results.
- **Attention Mechanisms**: Adding attention layers could help the model focus on important parts of the input sequence, potentially improving performance.
- **Hyperparameter Optimization**: Systematically exploring different hyperparameters (e.g., learning rate, batch size) might lead to better model performance.
- **Handling Unsupervised Data**: Leveraging the `train_unsup.csv` file through semi-supervised learning techniques could enhance the model's understanding of the data distribution.

## Training Steps

The training workflow followed these sequential steps:

1. **Environment Setup**: Installed necessary libraries and set up the environment for PyTorch and NLP processing.
2. **Data Loading**: Loaded the CSV files containing labeled movie reviews.
3. **Data Preprocessing**:
   - Tokenized the text data.
   - Built a vocabulary from the training data.
   - Numericalized the tokens based on the vocabulary.
4. **Dataset and DataLoader Creation**: Implemented custom classes to handle data batching and padding.
5. **Model Definition**: Defined the RNN architecture with LSTM layers.
6. **Model Training**:
   - Trained the model on the training set.
   - Validated the model on the validation set.
   - Saved the best model based on validation loss.
7. **Model Evaluation**: Assessed the model's performance on the test set.
8. **Visualization**: Plotted training and validation loss and accuracy over epochs.
9. **Confusion Matrix**: Generated a confusion matrix to visualize prediction performance.
10. **Custom Predictions**: Utilized the trained model to predict sentiments of new, unseen sentences.

## Conclusion

This RNN-based baseline serves as a foundational model for sentiment analysis on the IMDB dataset. By establishing this benchmark, it facilitates the comparison with more advanced models, such as Transformers with attention mechanisms, to evaluate improvements in performance and efficiency. Future work will involve implementing and training Transformer-based models to assess their superiority over the RNN baseline.

## Repository Structure

```
├── rnn_baseline_dataset
│   ├── train_neg.csv
│   ├── train_pos.csv
│   ├── test_neg.csv
│   ├── test_pos.csv
│   └── train_unsup.csv  # Not used in this baseline
├── rnn_baseline.py
└── README.md
```

- **`rnn_baseline_dataset/`**: Contains all the CSV files with labeled movie reviews.
- **`rnn_baseline.py`**: The main Python script implementing the RNN baseline model.
- **`README.md`**: This documentation file.
- **`requirements.txt`**: Lists all the Python dependencies required to run the project.

## Usage

To replicate the RNN baseline model:

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your_username/your_repository.git
   cd your_repository
   ```

2. **Install Dependencies**

   It's recommended to use a virtual environment.

   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the Dataset**

   Ensure that the `rnn_baseline_dataset` directory contains the necessary CSV files (`train_neg.csv`, `train_pos.csv`, `test_neg.csv`, `test_pos.csv`).

4. **Run the Model**

   Execute the Python script to train and evaluate the RNN model.

   ```bash
   python rnn_baseline.py
   ```

5. **View Results**

   - Training and validation loss and accuracy plots will be displayed.
   - The best-performing model will be saved as `best-model.pt`.
   - A confusion matrix will visualize the model's performance on the test set.
   - Custom sentiment predictions can be made by modifying the `sample_reviews` list in the script.

## License

This project is licensed under the [MIT License](LICENSE).

---

*Feel free to contribute to this project by submitting issues or pull requests. Your feedback and contributions are highly appreciated!*

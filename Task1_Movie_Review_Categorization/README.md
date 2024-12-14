# Movie Review Categorization

This project focuses on performing sentiment analysis on IMDB movie reviews using two distinct architectures:

1. **RNN Baseline (LSTM-based)**: A traditional approach leveraging Recurrent Neural Networks to model sequential data.
2. **Transformer Baseline**: A modern approach utilizing Transformer architectures with self-attention mechanisms to capture global dependencies within text.

By implementing and benchmarking these two models, we aim to understand the performance improvements and trade-offs between classic sequence models and state-of-the-art Transformer architectures in the context of sentiment analysis.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
  - [Dataset Preparation](#dataset-preparation)
  - [Dataset Files](#dataset-files)
- [Models](#models)
  - [RNN Baseline](#rnn-baseline)
  - [Transformer Baseline](#transformer-baseline)
- [Training Process](#training-process)
  - [Common Steps](#common-steps)
  - [RNN Training Steps](#rnn-training-steps)
  - [Transformer Training Steps](#transformer-training-steps)
- [Evaluation](#evaluation)
- [Potential Improvements](#potential-improvements)
- [Repository Structure](#repository-structure)
- [Usage](#usage)
  - [RNN Baseline](#rnn-baseline-usage)
  - [Transformer Baseline](#transformer-baseline-usage)
- [License](#license)

## Introduction

Sentiment analysis is a fundamental task in Natural Language Processing (NLP) that involves classifying text based on the expressed sentiment, typically as positive or negative. This project utilizes the IMDB movie reviews dataset to build and benchmark two different sentiment analysis models:

- **RNN (LSTM) Baseline**: Utilizes Recurrent Neural Networks with Long Short-Term Memory (LSTM) layers to capture sequential dependencies in text data.
- **Transformer Baseline**: Employs Transformer architectures with self-attention mechanisms to model long-range dependencies and contextual relationships within the text.

Establishing these baselines is crucial for comparing the effectiveness of different model architectures and for guiding future enhancements and optimizations.

## Dataset

### Dataset Preparation

The dataset used for this project is sourced from the [IMDB Movie Reviews](https://ai.stanford.edu/~amaas/data/sentiment/) dataset, curated by Stanford University. This dataset consists of 50,000 movie reviews labeled as either positive or negative, making it a suitable benchmark for binary sentiment classification tasks.

**Steps Taken:**

1. **Data Acquisition**: Downloaded the dataset from the official [IMDB website](https://ai.stanford.edu/~amaas/data/sentiment/).
2. **Data Conversion**: The original dataset comprises plain text files. To facilitate easier handling and processing within the models, the text files were manually converted into CSV format. This conversion streamlines data loading and preprocessing steps in the training pipeline.
3. **Data Organization**: The dataset is organized into separate CSV files for training and testing, further divided into positive and negative reviews.

### Dataset Files

All dataset files are located in the `movie_review_dataset` directory within this repository. The key files include:

- **`train_neg.csv`**: Contains negative movie reviews for training.
- **`train_pos.csv`**: Contains positive movie reviews for training.
- **`test_neg.csv`**: Contains negative movie reviews for testing.
- **`test_pos.csv`**: Contains positive movie reviews for testing.
- **`train_unsup.csv`**: Contains unlabeled movie reviews for unsupervised training (not used in this baseline).

**Note:** The `train_unsup.csv` file is included in the dataset but is not utilized in this baseline implementation. The primary reason for not using this file is to maintain a supervised learning setup, focusing solely on labeled data to establish clear performance benchmarks before introducing unsupervised or semi-supervised techniques.

## Models

### RNN Baseline

The RNN baseline employs a Recurrent Neural Network with LSTM layers, which are well-suited for capturing sequential dependencies in text data. The architecture comprises the following components:

1. **Embedding Layer**: Transforms input tokens into dense vector representations.
2. **LSTM Layers**: Capture the temporal dependencies in the sequence of embeddings.
3. **Fully Connected Layer**: Maps the LSTM outputs to the desired output dimension (binary classification).
4. **Dropout Layer**: Regularizes the model to prevent overfitting.

### Transformer Baseline

The Transformer baseline utilizes a Transformer-based architecture, which has demonstrated superior performance in various NLP tasks due to its ability to model long-range dependencies through self-attention mechanisms. The architecture includes:

1. **Embedding Layer**: Converts input tokens into dense vector representations.
2. **Positional Encoding**: Adds positional information to the embeddings to retain the order of tokens.
3. **Transformer Encoder Layers**: Consist of multi-head self-attention and feed-forward neural networks to process the embeddings.
4. **Classification Head**: Uses the representation of a special classification token (`<cls>`) to perform binary classification.
5. **Dropout Layer**: Applied for regularization.

## Training Process

### Common Steps

Both models follow a similar training workflow with shared preprocessing and data handling steps:

1. **Environment Setup**: Install necessary libraries and set up the environment for PyTorch and NLP processing.
2. **Data Loading**: Load the CSV files containing labeled movie reviews.
3. **Tokenization**: Use `spaCy` for efficient tokenization to convert text into tokens.
4. **Vocabulary Building**: Construct a vocabulary from the training data, mapping each unique token to an integer index.
5. **Dataset and DataLoader Creation**: Implement custom PyTorch `Dataset` and `DataLoader` classes to handle batching and padding of sequences.
6. **Model Initialization**: Initialize the respective model architectures with specified hyperparameters.
7. **Training**: Train the models for a set number of epochs, monitoring validation loss to save the best-performing model.
8. **Evaluation**: Evaluate the best model on the test set and record performance metrics.
9. **Visualization**: Visualize training and validation loss and accuracy over epochs.
10. **Prediction**: Use the trained models to predict sentiments of custom sentences.

### RNN Training Steps

The RNN training process involves:

1. **Model Definition**: Define the RNN architecture with LSTM layers, embedding, dropout, and a fully connected output layer.
2. **Loss and Optimizer**: Use Binary Cross-Entropy loss with logits and the Adam optimizer.
3. **Accuracy Metric**: Define a binary accuracy metric to track performance.
4. **Training Loop**: Train the model across multiple epochs, updating weights based on loss and optimizing for accuracy.
5. **Model Saving**: Save the best-performing model based on validation loss.

### Transformer Training Steps

The Transformer training process includes:

1. **Model Definition**: Define the Transformer-based architecture with embedding, positional encoding, Transformer encoder layers, and a classification head.
2. **Loss and Optimizer**: Utilize Binary Cross-Entropy loss with logits and the Adam optimizer.
3. **Accuracy Metric**: Implement a binary accuracy metric for performance tracking.
4. **Training Loop**: Train the Transformer model over several epochs, adjusting weights to minimize loss and maximize accuracy.
5. **Model Saving**: Save the best-performing Transformer model based on validation loss.

## Evaluation

Both models are evaluated using the following metrics:

- **Loss**: Binary Cross-Entropy loss to measure the discrepancy between predicted and actual labels.
- **Accuracy**: Percentage of correctly classified reviews.
- **Confusion Matrix**: Visual representation of true vs. predicted labels to understand model performance in detail.

Evaluation is conducted on a separate test set to assess the generalization capability of the trained models.

## Potential Improvements

While both baseline models provide solid foundations, several areas offer opportunities for enhancement:

- **Pre-trained Embeddings**: Incorporating pre-trained word embeddings like GloVe or Word2Vec could enhance the models' understanding of semantic relationships.
- **Bidirectional LSTM**: Further tuning of bidirectional LSTM parameters or experimenting with different architectures could improve results.
- **Advanced Attention Mechanisms**: Enhancing Transformer models with more sophisticated attention mechanisms might yield better performance.
- **Hyperparameter Optimization**: Systematic exploration of hyperparameters (e.g., learning rate, batch size) could lead to improved model performance.
- **Handling Unsupervised Data**: Leveraging the `train_unsup.csv` file through semi-supervised learning techniques could provide additional insights and improve model robustness.

## Dataset Repository
- **`CSC413-Final-Project/Task1_Movie_Review_Categorization/data`**: Contains all the CSV files with labeled movie reviews.
- **`rnn_baseline.ipynb`**: Jupyter Notebook implementing the RNN baseline model.
- **`transformer_baseline.ipynb`**: Jupyter Notebook implementing the Transformer baseline model.
- **`README.md`**: This documentation file.
- **`LICENSE`**: License information for the project.

## Usage

To replicate and experiment with both the RNN and Transformer baseline models, follow the steps below.

### RNN Baseline Usage

1. **Clone the Repository**

   ```bash
   git clone https://github.com/JoeZZG/CSC413-Final-Project.git
   cd CSC413-Final-Project
   ```

2. **Install Dependencies**

   It is recommended to use a virtual environment.

   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the Dataset**

   Ensure that the `movie_review_dataset` directory contains the necessary CSV files (`train_neg.csv`, `train_pos.csv`, `test_neg.csv`, `test_pos.csv`).

4. **Run the RNN Baseline Notebook**

   Open `rnn_baseline.ipynb` using Jupyter Notebook or Google Colab and execute all cells to train and evaluate the RNN model.

   ```bash
   jupyter notebook rnn_baseline.ipynb
   ```

5. **View Results**

   - Training and validation loss and accuracy plots will be displayed.
   - The best-performing model will be saved as `best-model.pt`.
   - A confusion matrix will visualize the model's performance on the test set.
   - Custom sentiment predictions can be made by modifying the `sample_reviews` list in the notebook.

### Transformer Baseline Usage

1. **Clone the Repository**

   If not already done, clone the repository as shown above.

2. **Install Dependencies**

   Ensure that dependencies are installed as per the `requirements.txt` file.

3. **Prepare the Dataset**

   Ensure that the `movie_review_dataset` directory contains the necessary CSV files (`train_neg.csv`, `train_pos.csv`, `test_neg.csv`, `test_pos.csv`).

4. **Run the Transformer Baseline Notebook**

   Open `transformer_baseline.ipynb` using Jupyter Notebook or Google Colab and execute all cells to train and evaluate the Transformer model.

   ```bash
   jupyter notebook transformer_baseline.ipynb
   ```

5. **View Results**

   - Training and validation loss and accuracy plots will be displayed.
   - The best-performing Transformer model will be saved as `best-transformer-model.pt`.
   - A confusion matrix will visualize the model's performance on the test set.
   - Custom sentiment predictions can be made by modifying the `sample_reviews` list in the notebook.

## License

This project is licensed under the [MIT License](LICENSE).

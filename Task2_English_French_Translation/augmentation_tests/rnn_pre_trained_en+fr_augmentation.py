import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
import nltk

# Uncomment if needed:
# nltk.download('punkt')

######################################################################
# Parameters
######################################################################
current_dir = os.getcwd()
data_dir = os.path.join(current_dir, "translation_train")
en_file = "../../data/small_vocab_en.txt"
fr_file = "../../data/small_vocab_fr.txt"

sample_max = 10000           # Maximum number of samples
batch_size = 64              # Batch size for training
embedding_size = 100         # Embedding dimension
hidden_size = 256            # Hidden size for the RNN
num_epochs = 40              # Number of training epochs
learning_rate = 0.001        # Learning rate for the optimizer
validation_split = 0.1       # Fraction of data used for validation
test_split = 0.1             # Fraction of data used for testing
max_sentence_length = 25     # Maximum sentence length (truncate/pad)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


######################################################################
# Data Loading and Preprocessing
######################################################################
def load_data(en_path: str, fr_path: str) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Load and tokenize English-French sentence pairs.

    Args:
        en_path (str): Path to English sentences file.
        fr_path (str): Path to French sentences file.

    Returns:
        Tuple[List[List[str]], List[List[str]]]: Tokenized English and French sentences.
    """
    with open(en_path, 'r', encoding='utf-8') as f_en:
        en_sentences = f_en.read().strip().split('\n')
    with open(fr_path, 'r', encoding='utf-8') as f_fr:
        fr_sentences = f_fr.read().strip().split('\n')

    data = list(zip(en_sentences, fr_sentences))
    random.shuffle(data)

    if sample_max < len(data):
        data = data[:sample_max]

    en_sentences, fr_sentences = zip(*data)
    en_tokenized = [s.strip().split() for s in en_sentences]
    fr_tokenized = [s.strip().split() for s in fr_sentences]
    return en_tokenized, fr_tokenized


def build_vocab(sentences: List[List[str]]) -> dict:
    """
    Build a vocabulary mapping tokens to unique indices, including special tokens.
    
    Special tokens:
        <pad>: padding
        <sos>: start of sequence
        <eos>: end of sequence
        <unk>: unknown token
    """
    vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
    for sent in sentences:
        for w in sent:
            if w not in vocab:
                vocab[w] = len(vocab)
    return vocab


def numericalize(sentences: List[List[str]], vocab: dict, max_len: int, add_eos: bool = False) -> torch.Tensor:
    """
    Convert tokenized sentences to indices, truncate/pad to fixed length, and optionally add <eos>.

    Args:
        sentences (List[List[str]]): Tokenized sentences.
        vocab (dict): Vocabulary mapping.
        max_len (int): Maximum length of sentences.
        add_eos (bool): Whether to append <eos> at the end of each sentence.

    Returns:
        torch.Tensor: Numericalized sentences as long tensors.
    """
    numer_data = []
    for sent in sentences:
        if add_eos:
            sent = sent + ["<eos>"]
        sent = sent[:max_len]
        numer_sent = [vocab.get(w, vocab["<unk>"]) for w in sent]
        while len(numer_sent) < max_len:
            numer_sent.append(vocab["<pad>"])
        numer_data.append(numer_sent)
    return torch.tensor(numer_data, dtype=torch.long)


class TranslationDataset(Dataset):
    """
    PyTorch Dataset for English-French translation.
    """
    def __init__(self, en_data: torch.Tensor, fr_data: torch.Tensor):
        self.en_data = en_data
        self.fr_data = fr_data

    def __len__(self) -> int:
        return len(self.en_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.en_data[idx], self.fr_data[idx]


######################################################################
# Model Definition (Seq2Seq with GRU RNN)
######################################################################
class EncoderRNN(nn.Module):
    """
    GRU-based encoder for seq2seq translation model.
    """
    def __init__(self, input_size: int, embed_size: int, hidden_size: int):
        super(EncoderRNN, self).__init__()
        self.embedding = nn.Embedding(input_size, embed_size, padding_idx=0)
        self.rnn = nn.GRU(embed_size, hidden_size, batch_first=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the encoder.

        Args:
            x (torch.Tensor): Source input batch (batch, src_len).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                outputs (batch, src_len, hidden_size) and hidden (1, batch, hidden_size).
        """
        embedded = self.embedding(x)
        outputs, hidden = self.rnn(embedded)
        return outputs, hidden


class DecoderRNN(nn.Module):
    """
    GRU-based decoder for seq2seq translation model.
    """
    def __init__(self, output_size: int, embed_size: int, hidden_size: int):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, embed_size, padding_idx=0)
        self.rnn = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the decoder.

        Args:
            x (torch.Tensor): Decoder input token (batch, 1).
            hidden (torch.Tensor): Decoder hidden state (1, batch, hidden_size).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                logits (batch, 1, output_size) and updated hidden state.
        """
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded, hidden)
        logits = self.fc(output)
        return logits, hidden


class Seq2Seq(nn.Module):
    """
    Seq2Seq model combining an encoder and a decoder for translation.
    """
    def __init__(self, encoder: EncoderRNN, decoder: DecoderRNN):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src: torch.Tensor, trg: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with teacher forcing.

        Args:
            src (torch.Tensor): Source input batch (batch, src_len).
            trg (torch.Tensor): Target input batch (batch, trg_len).

        Returns:
            torch.Tensor: Output logits of shape (batch, trg_len, tgt_vocab_size).
        """
        batch_size = src.size(0)
        trg_len = trg.size(1)
        trg_vocab_size = self.decoder.fc.out_features

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(device)

        # Encode
        enc_outputs, hidden = self.encoder(src)

        # First input to decoder is <sos>
        input_tok = torch.tensor([1] * batch_size, dtype=torch.long, device=device).unsqueeze(1)

        for t in range(trg_len):
            # Decoder step
            output, hidden = self.decoder(input_tok, hidden)
            outputs[:, t, :] = output.squeeze(1)
            # Teacher forcing: next input is the next token in the target sequence
            input_tok = trg[:, t].unsqueeze(1)

        return outputs


######################################################################
# Training & Evaluation Functions
######################################################################
def train_model(model: nn.Module, dataloader: DataLoader, criterion, optimizer) -> float:
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): Seq2Seq model.
        dataloader (DataLoader): Training data loader.
        criterion: Loss function.
        optimizer: Optimizer for parameter updates.

    Returns:
        float: Average training loss.
    """
    model.train()
    epoch_loss = 0
    for en_batch, fr_batch in dataloader:
        en_batch = en_batch.to(device)
        fr_batch = fr_batch.to(device)

        optimizer.zero_grad()
        output = model(en_batch, fr_batch)
        # Flatten outputs and targets
        output_dim = output.shape[-1]
        output = output.view(-1, output_dim)
        fr_batch = fr_batch.view(-1)

        loss = criterion(output, fr_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def evaluate_model(model: nn.Module, dataloader: DataLoader, criterion) -> float:
    """
    Evaluate the model on validation or test data.

    Args:
        model (nn.Module): Seq2Seq model.
        dataloader (DataLoader): Validation/test data loader.
        criterion: Loss function.

    Returns:
        float: Average evaluation loss.
    """
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for en_batch, fr_batch in dataloader:
            en_batch = en_batch.to(device)
            fr_batch = fr_batch.to(device)

            output = model(en_batch, fr_batch)
            output_dim = output.shape[-1]
            output = output.view(-1, output_dim)
            fr_batch = fr_batch.view(-1)

            loss = criterion(output, fr_batch)
            epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def translate_sentence(model: nn.Module, sentence: List[str], en_vocab: dict, fr_vocab: dict, max_len: int = 25) -> List[str]:
    """
    Translate an English sentence to French using greedy decoding.

    Args:
        model (nn.Module): Trained Seq2Seq model.
        sentence (List[str]): English input sentence tokens.
        en_vocab (dict): English vocabulary.
        fr_vocab (dict): French vocabulary.
        max_len (int): Maximum translation length.

    Returns:
        List[str]: Predicted French sentence tokens.
    """
    model.eval()

    en_indices = [en_vocab.get(w, en_vocab["<unk>"]) for w in sentence]
    en_indices = en_indices[:max_len]
    while len(en_indices) < max_len:
        en_indices.append(en_vocab["<pad>"])
    en_tensor = torch.tensor(en_indices, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        enc_outputs, hidden = model.encoder(en_tensor)
        input_tok = torch.tensor([fr_vocab["<sos>"]], dtype=torch.long, device=device).unsqueeze(1)

        predicted_tokens = []
        for _ in range(max_len):
            output, hidden = model.decoder(input_tok, hidden)
            pred_tok = output.argmax(2).item()
            if pred_tok == fr_vocab["<eos>"]:
                break
            predicted_tokens.append(pred_tok)
            input_tok = torch.tensor([pred_tok], dtype=torch.long, device=device).unsqueeze(1)

    inv_fr_vocab = {v: k for k, v in fr_vocab.items()}
    predicted_words = [inv_fr_vocab.get(t, "<unk>") for t in predicted_tokens]
    return predicted_words


def compute_bleu_score(model: nn.Module, dataloader: DataLoader, en_vocab: dict, fr_vocab: dict, num_samples: int = 100) -> float:
    """
    Compute the BLEU score on a subset of data.

    Args:
        model (nn.Module): Trained Seq2Seq model.
        dataloader (DataLoader): Data loader to compute BLEU on.
        en_vocab (dict): English vocabulary.
        fr_vocab (dict): French vocabulary.
        num_samples (int): Number of samples to use for BLEU computation.

    Returns:
        float: Computed BLEU score.
    """
    model.eval()
    references = []
    candidates = []
    inv_fr_vocab = {v: k for k, v in fr_vocab.items()}
    inv_en_vocab = {v: k for k, v in en_vocab.items()}

    count = 0
    with torch.no_grad():
        for en_batch, fr_batch in dataloader:
            en_batch = en_batch.to(device)
            fr_batch = fr_batch.to(device)

            batch_size = en_batch.size(0)
            for idx in range(batch_size):
                if count >= num_samples:
                    break
                src_seq = en_batch[idx].cpu().numpy().tolist()
                tgt_seq = fr_batch[idx].cpu().numpy().tolist()

                src_words = [inv_en_vocab.get(t, "<unk>") for t in src_seq if t != en_vocab["<pad>"]]
                tgt_words = [inv_fr_vocab.get(t, "<unk>") for t in tgt_seq if
                             t not in [fr_vocab["<pad>"], fr_vocab["<sos>"], fr_vocab["<unk>"], fr_vocab["<eos>"]]]

                pred_words = translate_sentence(model, src_words, en_vocab, fr_vocab, max_sentence_length)
                references.append([tgt_words])
                candidates.append(pred_words)
                count += 1

            if count >= num_samples:
                break

    bleu_score = nltk.translate.bleu_score.corpus_bleu(references, candidates)
    return bleu_score


def load_glove_embeddings(glove_path: str, vocab: dict, embed_dim: int) -> np.ndarray:
    """
    Load GloVe embeddings and create an embedding matrix for the given vocabulary.

    Args:
        glove_path (str): Path to the GloVe embeddings file.
        vocab (dict): Vocabulary mapping from token to index.
        embed_dim (int): Dimension of the GloVe embeddings.

    Returns:
        np.ndarray: Embedding matrix of shape (vocab_size, embed_dim).
                    If a word is not found, its embedding remains random.
    """
    # Initialize embedding matrix with random values
    embedding_matrix = np.random.normal(scale=0.1, size=(len(vocab), embed_dim)).astype(np.float32)

    # Load GloVe embeddings into a dictionary
    glove_dict = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vector = np.asarray(parts[1:], dtype=np.float32)
            if vector.shape[0] == embed_dim:
                glove_dict[word] = vector

    # Fill embedding_matrix with GloVe embeddings where possible
    for word, idx in vocab.items():
        if word in glove_dict:
            embedding_matrix[idx] = glove_dict[word]

    return embedding_matrix


######################################################################
# Main Script
######################################################################
if __name__ == "__main__":
    # Path to your pre-trained GloVe files
    glove_en_path = "../data/en_100d.txt" 
    glove_fr_path = "../data/fr_100d.txt"  

    # Load data
    en_path = os.path.join(data_dir, en_file)
    fr_path = os.path.join(data_dir, fr_file)
    en_sents, fr_sents = load_data(en_path, fr_path)

    # Build vocabularies
    en_vocab = build_vocab(en_sents)
    fr_vocab = build_vocab(fr_sents)

    # Numericalize data
    en_data = numericalize(en_sents, en_vocab, max_sentence_length, add_eos=False)
    fr_data = numericalize(fr_sents, fr_vocab, max_sentence_length, add_eos=True)

    # Split data into train, val, test
    total_len = len(en_data)
    val_len = int(total_len * validation_split)
    test_len = int(total_len * test_split)
    train_len = total_len - val_len - test_len

    indices = list(range(total_len))
    random.shuffle(indices)
    train_indices = indices[:train_len]
    val_indices = indices[train_len:train_len + val_len]
    test_indices = indices[train_len + val_len:]

    en_train, fr_train = en_data[train_indices], fr_data[train_indices]
    en_val, fr_val = en_data[val_indices], fr_data[val_indices]
    en_test, fr_test = en_data[test_indices], fr_data[test_indices]

    # Create Datasets and DataLoaders
    train_dataset = TranslationDataset(en_train, fr_train)
    val_dataset = TranslationDataset(en_val, fr_val)
    test_dataset = TranslationDataset(en_test, fr_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model
    input_size_en = len(en_vocab)
    output_size_fr = len(fr_vocab)
    encoder = EncoderRNN(input_size_en, embedding_size, hidden_size).to(device)
    decoder = DecoderRNN(output_size_fr, embedding_size, hidden_size).to(device)
    model = Seq2Seq(encoder, decoder).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=en_vocab["<pad>"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Load GloVe embeddings for English and French
    en_embedding_matrix = load_glove_embeddings(glove_en_path, en_vocab, embedding_size)
    fr_embedding_matrix = load_glove_embeddings(glove_fr_path, fr_vocab, embedding_size)

    with torch.no_grad():
        # Initialize encoder (English) and decoder (French) embedding layers with pre-trained GloVe embeddings
        model.encoder.embedding.weight.copy_(torch.from_numpy(en_embedding_matrix))
        model.decoder.embedding.weight.copy_(torch.from_numpy(fr_embedding_matrix))

    # Training loop
    train_losses = []
    val_losses = []
    val_bleu_scores = []

    for epoch in range(num_epochs):
        train_loss = train_model(model, train_loader, criterion, optimizer)
        val_loss = evaluate_model(model, val_loader, criterion)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Compute BLEU score on validation set
        val_bleu = compute_bleu_score(model, val_loader, en_vocab, fr_vocab, num_samples=100)
        val_bleu_scores.append(val_bleu)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Val BLEU Score: {val_bleu:.4f}")

    # Evaluate on test data
    test_loss = evaluate_model(model, test_loader, criterion)
    test_bleu = compute_bleu_score(model, test_loader, en_vocab, fr_vocab, num_samples=100)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"BLEU score on 100 test samples: {test_bleu:.4f}")

    # Append test metrics
    train_losses.append(None)  # No training loss for test
    val_losses.append(test_loss)
    val_bleu_scores.append(test_bleu)

    # Plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='blue')
    ax1.plot(range(1, num_epochs + 1), train_losses[:-1], label='Train Loss', color='blue')
    ax1.plot(range(1, num_epochs + 1), val_losses[:-1], label='Val Loss', color='cyan')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.set_ylabel('BLEU Score', color='red')
    ax2.plot(range(1, num_epochs + 1), val_bleu_scores[:-1], label='Val BLEU Score', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.legend(loc='upper right')

    plt.title('Training/Validation/Test Metrics vs. Epoch')
    plt.grid(True)

    # Display test metrics at the bottom of the plot
    # This text shows the final test loss and BLEU score.
    plt.figtext(0.5, 0.01, f"Test Loss: {test_loss:.4f}, Test BLEU: {test_bleu:.4f}",
                wrap=True, horizontalalignment='center', fontsize=12)

    plt.show()

    # Translate a random test sentence
    random_idx = random.randint(0, len(en_test) - 1)
    inv_en_vocab = {v: k for k, v in en_vocab.items()}
    inv_fr_vocab = {v: k for k, v in fr_vocab.items()}

    test_en_sentence = [inv_en_vocab[t] for t in en_test[random_idx].tolist() if t != en_vocab["<pad>"]]
    pred_fr = translate_sentence(model, test_en_sentence, en_vocab, fr_vocab, max_len=max_sentence_length)
    actual_fr_sentence = [inv_fr_vocab[t] for t in fr_test[random_idx].tolist() if t not in [fr_vocab["<pad>"], fr_vocab["<eos>"]]]

    print("English Input: ", " ".join(test_en_sentence))
    print("Model Translation: ", " ".join(pred_fr))
    print("Actual French: ", " ".join(actual_fr_sentence))

    # Custom Sentence Translation
    print("\n--- Custom Sentence Translation ---")
    custom_sentence = "I like apple."
    custom_tokens = custom_sentence.strip().split()
    pred_fr_custom = translate_sentence(model, custom_tokens, en_vocab, fr_vocab, max_len=max_sentence_length)
    print("English Input: ", custom_sentence)
    print("Model Translation: ", " ".join(pred_fr_custom))

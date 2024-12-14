import os
import random
import math
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
embedding_size = 128         # Embedding dimension
hidden_size = 256            # Hidden size for the Transformer
num_layers = 3               # Number of Transformer layers
num_heads = 8                # Number of attention heads
ffn_dim = 512                # Feedforward network dimension
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

    Args:
        sentences (List[List[str]]): Tokenized sentences.

    Returns:
        dict: Vocabulary mapping.
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
        max_len (int): Maximum sentence length.
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
# Model Definition (Transformer)
######################################################################
class PositionalEncoding(nn.Module):
    """
    Add positional information to embeddings using sine and cosine functions.
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)   # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)   # Odd indices
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to the input embeddings.

        Args:
            x (torch.Tensor): Input embeddings of shape (batch, seq_len, d_model).

        Returns:
            torch.Tensor: Embeddings with positional encoding added.
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class TransformerSeq2Seq(nn.Module):
    """
    Transformer-based sequence-to-sequence model for translation.
    """
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, d_model: int, nhead: int,
                 num_encoder_layers: int, num_decoder_layers: int, dim_feedforward: int,
                 max_len: int, dropout: float = 0.1):
        super(TransformerSeq2Seq, self).__init__()
        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=0)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)
        self.pos_decoder = PositionalEncoding(d_model, max_len=max_len)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward,
                                          batch_first=True, dropout=dropout)
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def make_src_key_padding_mask(self, src: torch.Tensor) -> torch.Tensor:
        """
        Create a mask for source padding tokens.

        Args:
            src (torch.Tensor): Source tensor of shape (batch, src_len).

        Returns:
            torch.Tensor: Padding mask of shape (batch, src_len).
        """
        return (src == 0)

    def make_tgt_key_padding_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        """
        Create a mask for target padding tokens.

        Args:
            tgt (torch.Tensor): Target tensor of shape (batch, tgt_len).

        Returns:
            torch.Tensor: Padding mask of shape (batch, tgt_len).
        """
        return (tgt == 0)

    def generate_square_subsequent_mask(self, size: int) -> torch.Tensor:
        """
        Generate a mask to prevent attention to subsequent positions.

        Args:
            size (int): Size of the mask.

        Returns:
            torch.Tensor: Subsequent mask of shape (size, size).
        """
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return mask.to(device)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Transformer model.

        Args:
            src (torch.Tensor): Source batch of shape (batch, src_len).
            tgt (torch.Tensor): Target batch of shape (batch, tgt_len).

        Returns:
            torch.Tensor: Logits of shape (batch, tgt_len, tgt_vocab_size).
        """
        src_padding_mask = self.make_src_key_padding_mask(src)
        tgt_padding_mask = self.make_tgt_key_padding_mask(tgt)
        tgt_seq_len = tgt.size(1)
        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len)

        src_emb = self.pos_encoder(self.src_embed(src))
        tgt_emb = self.pos_decoder(self.tgt_embed(tgt))

        output = self.transformer(src_emb, tgt_emb,
                                  src_key_padding_mask=src_padding_mask,
                                  tgt_key_padding_mask=tgt_padding_mask,
                                  tgt_mask=tgt_mask)
        logits = self.fc_out(output)
        return logits


######################################################################
# Training & Evaluation Functions
######################################################################
def train_model(model: nn.Module, dataloader: DataLoader, criterion, optimizer) -> float:
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): TransformerSeq2Seq model.
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

        # Target input is all but the last token; target output is all but the first token.
        tgt_input = fr_batch[:, :-1]
        tgt_output = fr_batch[:, 1:]

        optimizer.zero_grad()
        output = model(en_batch, tgt_input)           # (batch, tgt_len, vocab_size)
        output_dim = output.shape[-1]
        output = output.reshape(-1, output_dim)
        tgt_output = tgt_output.reshape(-1)

        loss = criterion(output, tgt_output)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    return epoch_loss / len(dataloader)


def evaluate_model(model: nn.Module, dataloader: DataLoader, criterion) -> float:
    """
    Evaluate the model on validation or test data.

    Args:
        model (nn.Module): TransformerSeq2Seq model.
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

            tgt_input = fr_batch[:, :-1]
            tgt_output = fr_batch[:, 1:]

            output = model(en_batch, tgt_input)
            output_dim = output.shape[-1]
            output = output.reshape(-1, output_dim)
            tgt_output = tgt_output.reshape(-1)

            loss = criterion(output, tgt_output)
            epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def translate_sentence(model: nn.Module, sentence: List[str], en_vocab: dict, fr_vocab: dict, max_len: int = 25) -> List[str]:
    """
    Translate an English sentence to French using greedy decoding.

    Args:
        model (nn.Module): Trained TransformerSeq2Seq model.
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
        # Start with <sos>
        tgt_indices = [fr_vocab["<sos>"]]
        for _ in range(max_len):
            tgt_tensor = torch.tensor(tgt_indices, dtype=torch.long, device=device).unsqueeze(0)
            output = model(en_tensor, tgt_tensor)
            next_token = output[0, -1].argmax().item()
            if next_token == fr_vocab["<eos>"]:
                break
            tgt_indices.append(next_token)

    inv_fr_vocab = {v: k for k, v in fr_vocab.items()}
    predicted_words = [inv_fr_vocab.get(t, "<unk>") for t in tgt_indices[1:]]
    return predicted_words


def compute_bleu_score(model: nn.Module, dataloader: DataLoader, en_vocab: dict, fr_vocab: dict, num_samples: int = 100) -> float:
    """
    Compute the BLEU score on a subset of the dataset.

    Args:
        model (nn.Module): Trained TransformerSeq2Seq model.
        dataloader (DataLoader): Data loader for BLEU computation.
        en_vocab (dict): English vocabulary.
        fr_vocab (dict): French vocabulary.
        num_samples (int): Number of samples to compute BLEU.

    Returns:
        float: Corpus BLEU score.
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
                tgt_words = [inv_fr_vocab.get(t, "<unk>") for t in tgt_seq if t not in [
                    fr_vocab["<pad>"], fr_vocab["<sos>"], fr_vocab["<unk>"], fr_vocab["<eos>"]]]

                pred_words = translate_sentence(model, src_words, en_vocab, fr_vocab, max_len=max_sentence_length)

                references.append([tgt_words])
                candidates.append(pred_words)
                count += 1

            if count >= num_samples:
                break

    bleu_score = nltk.translate.bleu_score.corpus_bleu(references, candidates)
    return bleu_score


######################################################################
# Main Script
######################################################################
if __name__ == "__main__":
    # Load data
    en_path = os.path.join(data_dir, en_file)
    fr_path = os.path.join(data_dir, fr_file)
    en_sents, fr_sents = load_data(en_path, fr_path)

    # Build vocabularies
    en_vocab = build_vocab(en_sents)
    fr_vocab = build_vocab(fr_sents)

    # Numericalize the data
    en_data = numericalize(en_sents, en_vocab, max_sentence_length, add_eos=False)
    fr_data = numericalize(fr_sents, fr_vocab, max_sentence_length, add_eos=True)

    # Pair the data for augmentation
    paired_data = list(zip(en_data, fr_data))

    # Split into long and short sentences based on English sentence length
    long_data = [pair for pair in paired_data if (pair[0] != 0).sum().item() > 15]
    short_data = [pair for pair in paired_data if (pair[0] != 0).sum().item() <= 15]

    # Calculate split sizes
    total_len = len(paired_data)
    val_len = int(total_len * validation_split)
    test_len = int(total_len * test_split)
    train_len = total_len - val_len - test_len

    # Determine the number of long and short sentences for training and validation
    train_long_count = int(0.2 * train_len)
    train_short_count = train_len - train_long_count

    val_long_count = int(0.2 * val_len)
    val_short_count = val_len - val_long_count

    # Shuffle long and short data
    random.shuffle(long_data)
    random.shuffle(short_data)

    # Sample training data
    train_long = long_data[:train_long_count]
    train_short = short_data[:train_short_count]
    train_data = train_long + train_short
    random.shuffle(train_data)

    # Sample validation data
    val_long = long_data[train_long_count:train_long_count + val_long_count]
    val_short = short_data[train_short_count:train_short_count + val_short_count]
    val_data = val_long + val_short
    random.shuffle(val_data)

    # Remaining data is for testing
    test_data = paired_data[train_len + val_len:]

    # Unzip the data
    en_train, fr_train = zip(*train_data) if train_data else ([], [])
    en_val, fr_val = zip(*val_data) if val_data else ([], [])
    en_test, fr_test = zip(*test_data) if test_data else ([], [])

    # Convert to tensors
    en_train = torch.stack(en_train) if en_train else torch.empty((0, max_sentence_length), dtype=torch.long)
    fr_train = torch.stack(fr_train) if fr_train else torch.empty((0, max_sentence_length), dtype=torch.long)
    en_val = torch.stack(en_val) if en_val else torch.empty((0, max_sentence_length), dtype=torch.long)
    fr_val = torch.stack(fr_val) if fr_val else torch.empty((0, max_sentence_length), dtype=torch.long)
    en_test = torch.stack(en_test) if en_test else torch.empty((0, max_sentence_length), dtype=torch.long)
    fr_test = torch.stack(fr_test) if fr_test else torch.empty((0, max_sentence_length), dtype=torch.long)

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

    model = TransformerSeq2Seq(
        src_vocab_size=input_size_en,
        tgt_vocab_size=output_size_fr,
        d_model=hidden_size,
        nhead=num_heads,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        dim_feedforward=ffn_dim,
        max_len=max_sentence_length
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)  
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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

    # After training, evaluate on the test set
    test_loss = evaluate_model(model, test_loader, criterion)
    test_bleu = compute_bleu_score(model, test_loader, en_vocab, fr_vocab, num_samples=1000)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"BLEU score on 1000 test samples: {test_bleu:.4f}")

    # Append test metrics as an additional data point
    train_losses.append(None)
    val_losses.append(test_loss)
    val_bleu_scores.append(test_bleu)

    # Plotting Training, Validation, and Test Metrics
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='blue')

    # Plot training and validation losses
    ax1.plot(range(1, num_epochs + 1), train_losses[:-1], label='Train Loss', color='blue')
    ax1.plot(range(1, num_epochs + 1), val_losses[:-1], label='Val Loss', color='cyan')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.legend(loc='upper left')

    # Plot test loss as text at the bottom of the plot
    # Comment: Display test loss and BLEU score at the bottom of the plot
    plt.figtext(0.5, 0.01, f"Test Loss: {test_loss:.4f}, Test BLEU: {test_bleu:.4f}",
                wrap=True, horizontalalignment='center', fontsize=12)

    # Create a twin Axes sharing the x-axis for BLEU scores
    ax2 = ax1.twinx()
    ax2.set_ylabel('BLEU Score', color='red')

    # Plot validation BLEU scores
    ax2.plot(range(1, num_epochs + 1), val_bleu_scores[:-1], label='Val BLEU Score', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.legend(loc='upper right')

    # Show the plot
    plt.title('Training/Validation/Test Metrics vs. Epoch')
    plt.grid(True)
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

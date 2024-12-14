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
data_dir = "C:\\Users\\Joe_Admin\\Desktop\\translation_train"
en_file = "small_vocab_en.txt"
fr_file = "small_vocab_fr.txt"

sample_max = 10000
batch_size = 64
embedding_size = 128
hidden_size = 256  # For transformer, this is the d_model dimension
num_layers = 3     # Number of encoder/decoder layers
num_heads = 8      # Number of attention heads
ffn_dim = 512      # Dimension of the feed-forward network
num_epochs = 20
learning_rate = 0.001
validation_split = 0.1
test_split = 0.1
max_sentence_length = 25

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


######################################################################
# Data Loading and Preprocessing
######################################################################
def load_data(en_path: str, fr_path: str) -> Tuple[List[List[str]], List[List[str]]]:
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


def build_vocab(sentences: List[List[str]]):
    vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
    for sent in sentences:
        for w in sent:
            if w not in vocab:
                vocab[w] = len(vocab)
    return vocab


def numericalize(sentences: List[List[str]], vocab: dict, max_len: int, add_eos: bool = False):
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
    def __init__(self, en_data, fr_data):
        self.en_data = en_data
        self.fr_data = fr_data

    def __len__(self):
        return len(self.en_data)

    def __getitem__(self, idx):
        return self.en_data[idx], self.fr_data[idx]


######################################################################
# Model Definition (Transformer)
######################################################################
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        seq_len = x.size(1)
        # Add positional encoding
        x = x + self.pe[:, :seq_len, :]
        return x


class TransformerSeq2Seq(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_len, dropout=0.1):
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

        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.d_model = d_model

    def make_src_key_padding_mask(self, src):
        # src: (batch, src_len)
        # mask: (batch, src_len) True where padding
        return (src == 0)

    def make_tgt_key_padding_mask(self, tgt):
        # tgt: (batch, tgt_len)
        return (tgt == 0)

    def generate_square_subsequent_mask(self, size):
        # Prevent the decoder from "seeing" tokens ahead in the sequence
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return mask.to(device)

    def forward(self, src, tgt):
        # src: (batch, src_len)
        # tgt: (batch, tgt_len)
        src_padding_mask = self.make_src_key_padding_mask(src)  # (batch, src_len)
        tgt_padding_mask = self.make_tgt_key_padding_mask(tgt)  # (batch, tgt_len)
        tgt_seq_len = tgt.size(1)

        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len)

        # Embeddings + positional encoding
        src_emb = self.pos_encoder(self.src_embed(src))  # (batch, src_len, d_model)
        tgt_emb = self.pos_decoder(self.tgt_embed(tgt))  # (batch, tgt_len, d_model)

        output = self.transformer(src_emb, tgt_emb,
                                  src_key_padding_mask=src_padding_mask,
                                  tgt_key_padding_mask=tgt_padding_mask,
                                  tgt_mask=tgt_mask)

        logits = self.fc_out(output)  # (batch, tgt_len, vocab_size)
        return logits


######################################################################
# Training & Evaluation Functions
######################################################################
def train_model(model, dataloader, criterion, optimizer):
    model.train()
    epoch_loss = 0
    for en_batch, fr_batch in dataloader:
        en_batch = en_batch.to(device)
        fr_batch = fr_batch.to(device)

        # The target input is the target sequence without the last token (since we predict next token)
        # The target output is the target sequence shifted by one (standard in transformer)
        tgt_input = fr_batch[:, :-1]  # all but last
        tgt_output = fr_batch[:, 1:]  # all but first

        optimizer.zero_grad()
        output = model(en_batch, tgt_input)
        # output: (batch, tgt_len-1, tgt_vocab_size)
        # tgt_output: (batch, tgt_len-1)
        output_dim = output.shape[-1]
        output = output.reshape(-1, output_dim)
        tgt_output = tgt_output.reshape(-1)

        loss = criterion(output, tgt_output)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    return epoch_loss / len(dataloader)


def evaluate_model(model, dataloader, criterion):
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


def translate_sentence(model, sentence: List[str], en_vocab, fr_vocab, max_len=25):
    model.eval()

    en_indices = [en_vocab.get(w, en_vocab["<unk>"]) for w in sentence]
    en_indices = en_indices[:max_len]
    while len(en_indices) < max_len:
        en_indices.append(en_vocab["<pad>"])
    en_tensor = torch.tensor(en_indices, dtype=torch.long, device=device).unsqueeze(0)  # (1, max_len)

    # Start with <sos>
    tgt_indices = [fr_vocab["<sos>"]]
    for i in range(max_len):
        tgt_tensor = torch.tensor(tgt_indices, dtype=torch.long, device=device).unsqueeze(0)
        # Generate predictions
        with torch.no_grad():
            output = model(en_tensor, tgt_tensor)  # (1, len(tgt_indices), vocab_size)
        next_token = output[0, -1].argmax().item()
        if next_token == fr_vocab["<eos>"]:
            break
        tgt_indices.append(next_token)

    inv_fr_vocab = {v: k for k, v in fr_vocab.items()}
    predicted_words = [inv_fr_vocab.get(t, "<unk>") for t in tgt_indices[1:]]  # skip <sos>
    return predicted_words


def compute_bleu_score(model, dataloader, en_vocab, fr_vocab, num_samples=100):
    model.eval()
    references = []
    candidates = []
    inv_fr_vocab = {v: k for k, v in fr_vocab.items()}
    inv_en_vocab = {v: k for k, v in en_vocab.items()}

    with torch.no_grad():
        count = 0
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
    en_path = os.path.join(data_dir, en_file)
    fr_path = os.path.join(data_dir, fr_file)
    en_sents, fr_sents = load_data(en_path, fr_path)

    en_vocab = build_vocab(en_sents)
    fr_vocab = build_vocab(fr_sents)

    en_data = numericalize(en_sents, en_vocab, max_sentence_length, add_eos=False)
    fr_data = numericalize(fr_sents, fr_vocab, max_sentence_length, add_eos=True)

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

    train_dataset = TranslationDataset(en_train, fr_train)
    val_dataset = TranslationDataset(en_val, fr_val)
    test_dataset = TranslationDataset(en_test, fr_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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

    criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore <pad> which is index 0
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []
    val_bleu_scores = []

    for epoch in range(num_epochs):
        train_loss = train_model(model, train_loader, criterion, optimizer)
        val_loss = evaluate_model(model, val_loader, criterion)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        val_bleu = compute_bleu_score(model, val_loader, en_vocab, fr_vocab, num_samples=100)
        val_bleu_scores.append(val_bleu)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Val BLEU Score: {val_bleu:.4f}")

    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', color='blue')
    ax1.plot(range(1, num_epochs + 1), val_losses, label='Val Loss', color='cyan')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('BLEU Score', color=color)
    ax2.plot(range(1, num_epochs + 1), val_bleu_scores, label='Val BLEU Score', color='red')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')

    plt.title('Training Loss and Validation BLEU Score vs. Epoch')
    plt.grid(True)
    plt.show()

    # Test evaluation
    test_loss = evaluate_model(model, test_loader, criterion)
    print(f"Test Loss: {test_loss:.4f}")

    bleu = compute_bleu_score(model, test_loader, en_vocab, fr_vocab, num_samples=100)
    print(f"BLEU score on 100 test samples: {bleu:.4f}")

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

    print("\n--- Custom Sentence Translation ---")
    custom_sentence = "I like apple."
    custom_tokens = custom_sentence.strip().split()
    pred_fr_custom = translate_sentence(model, custom_tokens, en_vocab, fr_vocab, max_len=max_sentence_length)
    print("English Input: ", custom_sentence)
    print("Model Translation: ", " ".join(pred_fr_custom))

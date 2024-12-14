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

# Ensure NLTK's BLEU scorer is ready
# Uncomment the following line if you haven't downloaded 'punkt' before
# nltk.download('punkt')

######################################################################
# Parameters
######################################################################
# Change these parameters as needed
data_dir = "C:\\Users\\Joe_Admin\\Desktop\\translation_train"
en_file = "small_vocab_en.txt"
fr_file = "small_vocab_fr.txt"

# Maximum number of samples to use (to speed up debugging)
sample_max = 10000  # Set smaller if you want quicker runs
batch_size = 64
embedding_size = 128
hidden_size = 256
num_epochs = 10  # Adjust number of epochs for training time
learning_rate = 0.001
validation_split = 0.1
test_split = 0.1
max_sentence_length = 25  # Truncate or pad sentences to a fixed length for ease

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
    # Pair them up
    data = list(zip(en_sentences, fr_sentences))
    # Shuffle data
    random.shuffle(data)
    if sample_max < len(data):
        data = data[:sample_max]
    en_sentences, fr_sentences = zip(*data)

    # Tokenize (simple split by space)
    en_tokenized = [s.strip().split() for s in en_sentences]
    fr_tokenized = [s.strip().split() for s in fr_sentences]
    return en_tokenized, fr_tokenized


def build_vocab(sentences: List[List[str]]):
    # Build vocabulary: add special tokens <pad>, <sos>, <eos>, <unk>
    vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
    for sent in sentences:
        for w in sent:
            if w not in vocab:
                vocab[w] = len(vocab)
    return vocab


def numericalize(sentences: List[List[str]], vocab: dict, max_len: int, add_eos: bool = False):
    # Convert tokens to indices
    # Truncate and pad
    # If add_eos=True, append <eos> token at the end
    numer_data = []
    for sent in sentences:
        if add_eos:
            sent = sent + ["<eos>"]
        # Truncate
        sent = sent[:max_len]
        # Convert
        numer_sent = [vocab.get(w, vocab["<unk>"]) for w in sent]
        # Pad if needed
        while len(numer_sent) < max_len:
            numer_sent.append(vocab["<pad>"])
        numer_data.append(numer_sent)
    # Ensure the dtype is compatible
    return torch.tensor(numer_data, dtype=torch.long)  # Changed from numpy array to torch tensor


class TranslationDataset(Dataset):
    def __init__(self, en_data, fr_data):
        self.en_data = en_data
        self.fr_data = fr_data

    def __len__(self):
        return len(self.en_data)

    def __getitem__(self, idx):
        return self.en_data[idx], self.fr_data[idx]


######################################################################
# Model Definition (Seq2Seq with RNN)
######################################################################
class EncoderRNN(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, embed_size, padding_idx=0)
        self.rnn = nn.GRU(embed_size, hidden_size, batch_first=True)

    def forward(self, x):
        embedded = self.embedding(x)
        # h_0 defaults to zeros if not provided
        outputs, hidden = self.rnn(embedded)
        return outputs, hidden


class DecoderRNN(nn.Module):
    def __init__(self, output_size, embed_size, hidden_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, embed_size, padding_idx=0)
        self.rnn = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        # x: (batch, 1)
        embedded = self.embedding(x)  # (batch, 1, embed_size)
        output, hidden = self.rnn(embedded, hidden)  # output: (batch, 1, hidden_size)
        logits = self.fc(output)  # (batch, 1, output_size)
        return logits, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg):
        # Teacher forcing approach
        batch_size = src.size(0)
        trg_len = trg.size(1)
        trg_vocab_size = self.decoder.fc.out_features

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(device)

        enc_outputs, hidden = self.encoder(src)

        # First input to decoder is <sos> token
        input_tok = torch.tensor([1] * batch_size, dtype=torch.long, device=device).unsqueeze(1)  # <sos>

        for t in range(trg_len):
            output, hidden = self.decoder(input_tok, hidden)
            outputs[:, t, :] = output.squeeze(1)
            # Teacher forcing
            input_tok = trg[:, t].unsqueeze(1)

        return outputs


######################################################################
# Training & Evaluation Functions
######################################################################
def train_model(model, dataloader, criterion, optimizer):
    model.train()
    epoch_loss = 0
    for en_batch, fr_batch in dataloader:
        en_batch = en_batch.to(device)
        fr_batch = fr_batch.to(device)

        optimizer.zero_grad()
        output = model(en_batch, fr_batch)
        # output: (batch, trg_len, trg_vocab_size)
        # fr_batch: (batch, trg_len)
        # Shift fr_batch by one for target? Since we appended <eos> at the end,
        # we can predict each token except the first <sos>. Here we just predict
        # target tokens directly (assuming fr_batch includes <eos> as last token).

        # Flatten for cross entropy
        output_dim = output.shape[-1]
        output = output.view(-1, output_dim)
        fr_batch = fr_batch.view(-1)

        loss = criterion(output, fr_batch)
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
            output = model(en_batch, fr_batch)
            output_dim = output.shape[-1]
            output = output.view(-1, output_dim)
            fr_batch = fr_batch.view(-1)
            loss = criterion(output, fr_batch)
            epoch_loss += loss.item()
    return epoch_loss / len(dataloader)


def translate_sentence(model, sentence: List[str], en_vocab, fr_vocab, max_len=25):
    # Convert sentence to indices
    en_indices = [en_vocab.get(w, en_vocab["<unk>"]) for w in sentence]
    en_indices = en_indices[:max_len]
    while len(en_indices) < max_len:
        en_indices.append(en_vocab["<pad>"])
    en_tensor = torch.tensor(en_indices, dtype=torch.long, device=device).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        # Encode
        enc_outputs, hidden = model.encoder(en_tensor)
        # Start with <sos>
        input_tok = torch.tensor([1], dtype=torch.long, device=device).unsqueeze(1)  # <sos>

        predicted_tokens = []
        for _ in range(max_len):
            output, hidden = model.decoder(input_tok, hidden)
            # Pick top token
            pred_tok = output.argmax(2).item()
            if pred_tok == fr_vocab["<eos>"]:
                break
            predicted_tokens.append(pred_tok)
            input_tok = torch.tensor([pred_tok], dtype=torch.long, device=device).unsqueeze(1)

    # Convert tokens back to words
    inv_fr_vocab = {v: k for k, v in fr_vocab.items()}
    predicted_words = [inv_fr_vocab.get(t, "<unk>") for t in predicted_tokens]
    return predicted_words


def compute_bleu_score(model, dataloader, en_vocab, fr_vocab, num_samples=100):
    # Simple BLEU evaluation on a subset of the data
    model.eval()
    references = []
    candidates = []
    inv_fr_vocab = {v: k for k, v in fr_vocab.items()}

    with torch.no_grad():
        for i, (en_batch, fr_batch) in enumerate(dataloader):
            en_batch = en_batch.to(device)
            fr_batch = fr_batch.to(device)
            # We'll just do greedy decoding similar to translate_sentence for each
            # sample in the batch
            batch_size = en_batch.size(0)
            for idx in range(batch_size):
                if len(references) >= num_samples:
                    break
                src_seq = en_batch[idx].cpu().numpy().tolist()
                tgt_seq = fr_batch[idx].cpu().numpy().tolist()

                # Convert src_seq to words
                inv_en_vocab = {v: k for k, v in en_vocab.items()}
                src_words = [inv_en_vocab.get(t, "<unk>") for t in src_seq if t not in [en_vocab["<pad>"]]]

                # Convert tgt_seq to words (remove <pad>, <eos>, <sos>, <unk>)
                tgt_words = [inv_fr_vocab.get(t, "<unk>") for t in tgt_seq if
                             t not in [fr_vocab["<pad>"], fr_vocab["<sos>"], fr_vocab["<unk>"], fr_vocab["<eos>"]]]

                # Translate
                pred_words = translate_sentence(model, src_words, en_vocab, fr_vocab, max_sentence_length)

                references.append([tgt_words])  # BLEU expects list of references
                candidates.append(pred_words)

            if len(references) >= num_samples:
                break

    # Compute BLEU
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

    # Build vocabulary
    en_vocab = build_vocab(en_sents)
    fr_vocab = build_vocab(fr_sents)

    # Numericalize
    en_data = numericalize(en_sents, en_vocab, max_sentence_length, add_eos=False)
    fr_data = numericalize(fr_sents, fr_vocab, max_sentence_length, add_eos=True)  # French has <eos>

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

    train_dataset = TranslationDataset(en_train, fr_train)
    val_dataset = TranslationDataset(en_val, fr_val)
    test_dataset = TranslationDataset(en_test, fr_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    input_size_en = len(en_vocab)
    output_size_fr = len(fr_vocab)
    encoder = EncoderRNN(input_size_en, embedding_size, hidden_size).to(device)
    decoder = DecoderRNN(output_size_fr, embedding_size, hidden_size).to(device)
    model = Seq2Seq(encoder, decoder).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=en_vocab["<pad>"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training
    train_losses = []
    val_losses = []
    val_bleu_scores = []  # List to store BLEU scores per epoch

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

    # Visualization of training loss and validation BLEU score
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', color='blue')
    ax1.plot(range(1, num_epochs + 1), val_losses, label='Val Loss', color='cyan')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:red'
    ax2.set_ylabel('BLEU Score', color=color)  # we already handled the x-label with ax1
    ax2.plot(range(1, num_epochs + 1), val_bleu_scores, label='Val BLEU Score', color='red')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')

    plt.title('Training Loss and Validation BLEU Score vs. Epoch')
    plt.grid(True)
    plt.show()

    # Evaluate on test data
    test_loss = evaluate_model(model, test_loader, criterion)
    print(f"Test Loss: {test_loss:.4f}")

    # Compute BLEU score on a small subset of test data
    bleu = compute_bleu_score(model, test_loader, en_vocab, fr_vocab, num_samples=100)
    print(f"BLEU score on 100 test samples: {bleu:.4f}")

    # Try translating a random sentence from the test set
    random_idx = random.randint(0, len(en_test) - 1)

    # Convert the tensors to lists before accessing the vocabulary
    inv_en_vocab = {v: k for k, v in en_vocab.items()}
    inv_fr_vocab = {v: k for k, v in fr_vocab.items()}

    # Option 1: Using .tolist()
    test_en_sentence = [inv_en_vocab[t] for t in en_test[random_idx].tolist() if t != en_vocab["<pad>"]]

    # **Alternatively, Option 2: Using t.item()**
    # test_en_sentence = [inv_en_vocab[t.item()] for t in en_test[random_idx] if t.item() != en_vocab["<pad>"]]

    # Translate
    pred_fr = translate_sentence(model, test_en_sentence, en_vocab, fr_vocab, max_len=max_sentence_length)

    # Prepare actual French sentence
    # Similarly, ensure tensors are converted to integers
    actual_fr_sentence = [inv_fr_vocab[t] for t in fr_test[random_idx].tolist() if
                          t not in [fr_vocab["<pad>"], fr_vocab["<eos>"]]]

    # Print out
    print("English Input: ", " ".join(test_en_sentence))
    print("Model Translation: ", " ".join(pred_fr))
    print("Actual French: ", " ".join(actual_fr_sentence))

    # **Added Code for Translating "I like apple."**
    print("\n--- Custom Sentence Translation ---")
    custom_sentence = "I like apple."
    # Tokenize the custom sentence (same as training)
    custom_tokens = custom_sentence.strip().split()
    # Translate
    pred_fr_custom = translate_sentence(model, custom_tokens, en_vocab, fr_vocab, max_len=max_sentence_length)
    # Print the translation
    print("English Input: ", custom_sentence)
    print("Model Translation: ", " ".join(pred_fr_custom))

    # Explanation of evaluation:
    # We used a simple loss-based evaluation on validation and test sets.
    # Additionally, we computed BLEU score for a subset of the test set,
    # which is a standard metric for evaluating machine translation quality.
    # BLEU compares the overlap of n-grams between the predicted translation
    # and the reference translation.

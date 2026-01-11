"""Tiny char-level RNN with PyTorch trained on a very small corpus.
This is intentionally tiny and not meant for serious generation — it's a demo.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import random


class CharRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size=64):
        super().__init__()
        self.hidden_size = hidden_size
        self.emb = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h=None):
        x = self.emb(x)
        out, h = self.rnn(x, h)
        logits = self.fc(out)
        return logits, h


def build_vocab(text: str):
    chars = sorted(list(set(text)))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    return stoi, itos


def encode(text: str, stoi):
    return [stoi[c] for c in text]


def decode(seq, itos):
    return "".join(itos[int(i)] for i in seq)


def train_and_generate(corpus: str = "hello world\n", epochs: int = 200, gen_len: int = 100):
    stoi, itos = build_vocab(corpus)
    vocab_size = len(stoi)
    model = CharRNN(vocab_size)
    optimz = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    data = encode(corpus, stoi)
    seq = torch.tensor(data, dtype=torch.long).unsqueeze(0)  # batch size 1

    for epoch in range(epochs):
        model.train()
        optimz.zero_grad()
        logits, _ = model(seq)
        # shift targets
        logits = logits[:, :-1, :].reshape(-1, vocab_size)
        targets = seq[:, 1:].reshape(-1)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimz.step()
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs} loss={loss.item():.4f}")

    # generation
    model.eval()
    with torch.no_grad():
        input_idx = torch.tensor([[data[0]]], dtype=torch.long)
        hidden = None
        out_seq = [data[0]]
        for _ in range(gen_len - 1):
            logits, hidden = model(input_idx, hidden)
            probs = torch.softmax(logits[:, -1, :], dim=-1).cpu().numpy().ravel()
            next_idx = int(random.choices(range(vocab_size), weights=probs, k=1)[0])
            out_seq.append(next_idx)
            input_idx = torch.tensor([[next_idx]], dtype=torch.long)
    return decode(out_seq, itos)


if __name__ == "__main__":
    sample = train_and_generate()
    print("Generated (tiny) text:\n", sample)

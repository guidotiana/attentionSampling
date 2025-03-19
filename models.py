import torch
import torch.nn as nn
import torch.nn.functional as F


class Embedding(nn.Module):
    def __init__(self, vocab_size, d):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d)

    def forward(self, x):
        return self.embedding(x)




class PositionalEncoding(nn.Module):
    def __init__(self, d, n, device):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(n, d, device=device)
        position = torch.arange(0, n, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d, 2, dtype=torch.float, device=device) * (-torch.log(torch.tensor(10000.0, device=device)) / d))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        seq_len = x.size(1)
        return self.encoding[:, :seq_len, :]




class MultiHeadAttention(nn.Module):
    def __init__(self, d, H):
        super(MultiHeadAttention, self).__init__()
        self.H = H
        self.d = d
        self.k = d // H

        assert self.k * H == d, "d must be divisible by H"

        self.Wq = nn.Linear(d, d, bias=False)
        self.Wk = nn.Linear(d, d, bias=False)
        self.Wv = nn.Linear(d, d, bias=False)
        self.Wc = nn.Linear(d, d, bias=False)

    def forward(self, x, mask=None):
        N, n, d = x.size()
        Q = self.Wq(x).view(N, n, self.H, self.k).transpose(1, 2)
        K = self.Wk(x).view(N, n, self.H, self.k).transpose(1, 2)
        V = self.Wv(x).view(N, n, self.H, self.k).transpose(1, 2)
        scores = torch.einsum("nhql,nhkl->nhqk", [Q, K]) / (self.k ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(scores, dim=-1)
        out = torch.einsum("nhql,nhlv->nhqv", [attention, V])
        out = out.transpose(1, 2).contiguous().view(N, n, d)
        return self.Wc(out)




class FeedForward(nn.Module):
    def __init__(self, d, m):
        super(FeedForward, self).__init__()
        self.W1 = nn.Linear(d, m)
        self.W2 = nn.Linear(m, d)

    def forward(self, x):
        return self.W2(F.relu(self.W1(x)))




class TransformerBlock(nn.Module):
    def __init__(self, d, H, m, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(d, H)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ff = FeedForward(d, m)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_out = self.dropout(self.attention(x, mask))
        x = self.norm1(x + attn_out)
        ff_out = self.dropout(self.ff(x))
        x = self.norm2(x + ff_out)
        return x




class Transformer(nn.Module):
    def __init__(self, vocab_size, d, H, m, L, n, device, dropout=0.1):
        super(Transformer, self).__init__()
        self.token_embedding = Embedding(vocab_size, d)
        self.positional_encoding = PositionalEncoding(d, n, device)
        self.layers = nn.ModuleList([
            TransformerBlock(d, H, m, dropout) for _ in range(L)
        ])
        self.dropout = nn.Dropout(dropout)
        self.device = device

    def forward(self, x, mask=None):
        x = x.to(self.device)
        x = self.token_embedding(x) + self.positional_encoding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x




class Model(nn.Module):
    def __init__(self, transformer, d, vocab_size, device):
        super().__init__()
        self.transformer = transformer
        self.linear = nn.Linear(d, vocab_size)
        self.softmax = nn.Softmax(dim=-1)
        self.device = device

    def forward(self, x):
        x = self.transformer(x)
        return self.linear(x)




# transformer = Transformer(VOCAB_SIZE, d, H, m, L, n, device)
# model = Model(transformer, d, VOCAB_SIZE, device)


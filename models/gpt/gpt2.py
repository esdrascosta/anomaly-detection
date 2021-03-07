# https://github.com/teddykoker/image-gpt

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(Block, self).__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x):
        attn_mask = torch.full(
            (len(x), len(x)), -float("Inf"), device=x.device, dtype=x.dtype
        )
        attn_mask = torch.triu(attn_mask, diagonal=1)

        x = self.ln_1(x)
        a, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        return x


class GPT2(nn.Module):
    def __init__(
        self, embed_dim, num_heads, num_layers, num_positions, num_vocab, num_classes
    ):
        super(GPT2, self).__init__()

        self.embed_dim = embed_dim

        # start of sequence token
        self.sos = torch.nn.Parameter(torch.zeros(embed_dim))
        nn.init.normal_(self.sos)

        self.token_embeddings = nn.Embedding(num_vocab, embed_dim)
        self.position_embeddings = nn.Embedding(num_positions, embed_dim)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(Block(embed_dim, num_heads))

        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_vocab, bias=False)
        self.clf_head = nn.Linear(embed_dim, num_classes)

    def forward(self, x, classify=False):

        x = self._shape_input(x)

        """
        Expect input as shape [sequence len, batch]
        If classify, return classification logits
        """

        length, batch = x.shape

        h = self.token_embeddings(x)

        # prepend sos token
        sos = torch.ones(1, batch, self.embed_dim, device=x.device) * self.sos
        h = torch.cat([sos, h[:-1, :, :]], axis=0)

        # add positional embeddings
        positions = torch.arange(length, device=x.device).unsqueeze(-1)
        h = h + self.position_embeddings(positions).expand_as(h)

        # transformer
        for layer in self.layers:
            h = layer(h)

        if not classify:
            # return logits
            return self.head(h)

        h = torch.mean(h, dim=0)  # average pool over sequence
        return self.clf_head(h)  # return classification logits

    def _shape_input(self, x):
        """shape batch of images for input into GPT2 model"""
        x = x.view(x.shape[0], -1)  # flatten images into sequences
        x = x.transpose(0, 1).contiguous()  # to shape [seq len, batch]
        return x


if __name__ == '__main__':
    device = 'cpu'
    sample = torch.rand(1, 1, 28, 28).to(device).to(torch.int64)
    print(f'sample={sample.view(-1).size()}')
    model = GPT2(
        embed_dim=16,
        num_heads=2,
        num_layers=8,
        num_positions=28*28,
        num_vocab=16,
        num_classes=10
    ).to(device)
    result = model(sample)

    print(f'result={result.view(-1, result.size(-1)).size()}')

    print(f"result={result.size()}")

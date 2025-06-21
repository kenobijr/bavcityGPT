import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

""" core logic / classes to setup transformer nn"""


@dataclass
class GPTconfig:
    """
    - configuration class for model architecture hyperparameters
    - mandatory as input to instanciate core GTP class
    - training & sampling config params in separate files in config dir
    """
    context_len: int = 64  # block_size
    vocab_size: int = 61
    n_embd: int = 256
    n_head: int = 8
    n_layer: int = 8
    dropout: float = 0.2
    a_bias: bool = True  # bias true for q, k, v, proj in attention layers
    ffw_bias: bool = True  # bias true for lin layers in ffw modules, since they follow layernorm layers
    lm_head_bias: bool = False


class Head(nn.Module):
    """
    - single self-attention head; 
    - called from multi-head-attention class
    - pre-registered full-size buffer for triangular masking
    """
    
    def __init__(self, config: GPTconfig, h_size: int):
        super().__init__()
        self.query = nn.Linear(config.n_embd, h_size, bias=config.a_bias)
        self.key = nn.Linear(config.n_embd, h_size, bias=config.a_bias)
        self.value = nn.Linear(config.n_embd, h_size, bias=config.a_bias)
        # helper matrix for triangular masking; all zero values above diagonal
        self.register_buffer("tril", torch.tril(torch.ones(config.context_len, config.context_len)))
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x) -> torch.Tensor:
        B, T, C = x.shape
        # B, T, H
        q = self.query(x)
        k = self.key(x)
        # B, T, T
        wei = q @ torch.transpose(k, dim0=-1, dim1=-2) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.drop(wei)
        # B, T, H
        v = self.value(x)
        out = wei @ v
        return out
    

class MultiHeadAttention(nn.Module):
    """
    - steering multiple heads of self-attention in parallel
    - n_embd / n_head must have no remainder
    """
    
    def __init__(self, config:GPTconfig, n_head, head_size):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.head_size: int = config.n_embd // config.n_head
        self.heads = nn.ModuleList(Head(config, self.head_size) for _ in range(config.n_head))
        # linear projection layer to blend all cat head outputs
        self.proj = nn.Linear(n_embd, n_embd, bias=config.a_bias)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # cat / stack each head's out_features along last dim to total of n_embd out_features
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.drop(self.proj(out))
        return out
    

# mlp layer with relu; widened first linear layer
class Ffw(nn.Module):
    
    def __init__(self, config:GPTconfig):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4, bias=config.ffw_bias),
            nn.ReLU(),
            nn.Linear(n_embd * 4, n_embd, bias=config.ffw_bias),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.layers(x)
    

# transformer block: communication in multi-head-attention, then computation in ffw layers
class TransformerBlock(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.multi_head_sa = MultiHeadAttention(config)
        self.ffw = Ffw()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        x = x + self.multi_head_sa((self.ln1(x)))
        x = x + self.ffw(self.ln2(x))
        return x
    

# Core GPT logic setting up NN
class GPT(nn.Module):
    
    def __init__(self, config:GPTconfig):
        super().__init__()
        # embeddings
        self.tok_embeddings = nn.Embedding(vocab_size, n_embd)
        self.pos_embeddings = nn.Embedding(context_len, n_embd)
        # transformer blocks of amount n_layer
        self.t_blocks = nn.Sequential(*[TransformerBlock() for _ in range(n_layer)])
        # output layer
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=config.lm_head_bias)
 
    def forward(self, idx, targets=None):
        B, T = idx.shape
        # x comes as B, T; token embeddings; B,T,C
        tok_emb = self.tok_embeddings(idx)
        # creates 1D-tensor with values from 0 - context_len; T
        pos_idx = torch.arange(0, T, device=device)
        # position embeddings; T, C
        pos_emb = self.pos_embeddings(pos_idx)
        # combined emds for token + pos; B, T, C
        emb = tok_emb + pos_emb
        # hidden layers & logits
        h = self.t_blocks(emb)
        logits = self.lm_head(h)
        # calc loss if targets are available; otherwise set loss to None for sampling
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            # flatten logits into B*T, C
            logits = logits.view(B*T, C)
            # flatten targets into B*T
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    # generate names of tbd amount; name ends at first line break char
    def generate(self, amount_names):
        out = []
        for _ in range(amount_names):
            name = []
            # start always with 0 context for linebreak as first char; forward pass expects shape of (1, 1) to work
            context = torch.zeros((1, 1), dtype=torch.long)
            context = context.to(device)
            while True:
                # context must not be greater than context_len, otherwise mat mul in forward pass does not work; cut max latest context
                context_cut = context[:, -context_len:]
                logits, _ = self(context_cut)
                # grab logits at last timestep
                logits = logits[:, -1, :]
                logits = F.softmax(logits, dim=-1)
                idx = torch.multinomial(logits, num_samples=1, replacement=True).item()
                name.append(itos[idx])
                # end name gen when first linebreak is sampled
                if idx == 0:
                    break
                else:
                    # as long as no linebreak is hit, add last idx to context and sample next char for name
                    context = torch.cat((context, torch.tensor([[idx]], dtype=torch.long, device=device)), dim=1)
            out.append("".join(name))
        return out
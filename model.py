import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple

""" Bavarian City Name GPT // core classes to setup transformer nn"""


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
    ffw_widen: int = 4  # factor to widen linear layer in ffw module
    a_bias: bool = True  # bias true for q, k, v, proj in attention layers
    ffw_bias: bool = True  # bias true for lin layers in ffw modules;
    lm_head_bias: bool = False


class MultiHeadAttention(nn.Module):
    """  """

    def __init__(self, config: GPTconfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "Ratio n_embd / n_head must have no remainder."
        self.n_head = config.n_head
        self.head_size: int = config.n_embd // config.n_head
        # single layer for all heads; 3 is constant
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.a_bias)
        # linear projection layer to blend all cat head outputs
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.a_bias)
        self.dropout = nn.Dropout(config.dropout)
        # helper matrix for triangular masking; all zero values above diagonal
        self.register_buffer("tril", torch.tril(torch.ones(config.context_len, config.context_len)))

    def forward(self, x) -> torch.Tensor:
        B, T, C = x.shape
        # permute qkv-dim to front to catch qkv in next step
        qkv = self.qkv(x).reshape(B, T, 3, self.n_head, self.head_size).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, nh, T, hs
        att = (q @ torch.transpose(k, dim0=-1, dim1=-2)) * self.head_size**-0.5  # B,nh,T, T
        # = attention weights
        att = F.softmax(att.masked_fill(self.tril[:T, :T] == 0, float("-inf")), dim=-1)
        # att @ v (=attended values): B,nh,T,hs
        out = (self.dropout(att) @ v).transpose(1, 2).reshape(B, T, C)
        return self.dropout(self.proj(out))


class Ffw(nn.Module):
    """
    - mlp layer with relu non-linearity
    - widened by tbd factor; dropout before returning output
    - explicit layer naming to catch proj weights for weight init
    """

    def __init__(self, config: GPTconfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * config.ffw_widen, bias=config.ffw_bias)
        self.relu = nn.ReLU()
        self.proj = nn.Linear(config.n_embd * config.ffw_widen, config.n_embd, bias=config.ffw_bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.relu(x)
        x = self.proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    - communication in multi-head-attention, computation in ffw layers
    - layernorm -> attention -> skip -> layernorm -> ffw -> skip
    """

    def __init__(self, config: GPTconfig):
        super().__init__()
        self.multi_head_sa = MultiHeadAttention(config)
        self.ffw = Ffw(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x) -> torch.Tensor:
        x = x + self.multi_head_sa((self.ln1(x)))
        x = x + self.ffw(self.ln2(x))
        return x


# Core GPT logic setting up NN
class GPT(nn.Module):
    """central class setting up the NN; flag to deactivate _init_weights for testcases"""

    def __init__(self, config: GPTconfig, init_weights: bool = True):
        super().__init__()
        assert isinstance(config, GPTconfig), "Invalid config type."
        assert config.n_head > 0, "n_head must be positive."
        assert config.n_layer > 0, "n_layer must be positive."
        assert config.vocab_size > 0, "vocab_size must be positive."
        self.config = config
        # embeddings & transformer blocks
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.context_len, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        # output layer
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=config.lm_head_bias)
        # weight tying between token embedding and output projection
        self.transformer.wte.weight = self.lm_head.weight
        # trigger weight init
        if init_weights:
            self.apply(self._init_weights)

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.tensor]]:
        B, T = idx.shape
        assert T <= self.config.context_len, f"T: {T} exceeds context_len {self.config.context_len}"
        x = self.transformer.drop(
            self.transformer.wte(idx) +
            self.transformer.wpe(torch.arange(T, dtype=torch.long, device=idx.device))
        )
        # forward through hidden layers & layernorm
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        # calc loss if targets are available; otherwise loss is None
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :])  # list [-1] to preserve the time dim
            loss = None
        return logits, loss

    def _init_weights(self, module) -> None:
        """standard initfor lin / embd layers; scaled residual init for projection layers"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # init weights with small normal distribution (GPT-2 standard)
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if hasattr(module, "bias") and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        # scaled init to residual projections
        for name, param in self.named_parameters():
            # catches both attention and ffw projections
            if name.endswith("proj.weight"):
                torch.nn.init.normal_(
                    param, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layer)
                )

    def get_num_params(self, non_embedding=True):
        """
        - return the number of parameters in the model.
        - For non-embedding count (default), the pos embeddings get subtracted
        - the token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

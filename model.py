import torch
import torch.nn as nn
from typing import Tuple
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embed: int = 768
    dropout: float = 0.0
    n_hidden: int = 1024
    bias: bool = True
    tie_weights: bool = True
    pad_token_id: int = -100

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class Block(nn.Module):
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.ln_1 = LayerNorm(config.n_embed, bias=config.bias)
        self.attn = nn.MultiheadAttention(config.n_embed, config.n_head, dropout=config.dropout, batch_first=True)
        self.ln_2 = LayerNorm(config.n_embed, bias=config.bias)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embed, config.n_hidden),
            nn.GELU(),
            nn.Linear(config.n_hidden, config.n_embed),
            nn.Dropout(config.dropout),
        )

    def forward(self, x, mask):
        x = x + self.attn(self.ln_1(x), self.ln_1(x), self.ln_1(x), key_padding_mask=mask)[0]
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embed)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embed)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embed, bias=config.bias)
        self.head = nn.Linear(config.n_embed, config.vocab_size, bias=config.bias)

        if config.tie_weights:
            self.head.weight = self.tok_emb.weight

        # init all weights
        self.apply(self._init_weights)

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self):
        """
        Return the number of parameters in the model.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if self.config.tie_weights:
            n_params -= self.tok_emb.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, mask, targets=None) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.tok_emb(x) + self.pos_emb(torch.arange(x.shape[1], device=x.device)) # N, L, E
        for blk in self.blocks:
            x = blk(x, mask)
        x = self.ln_f(x)
        logits = self.head(x)

        if targets is not None:
            loss = F.cross_entropy(logits.contiguous().view(-1, logits.size(-1)), targets.contiguous().view(-1), ignore_index=self.config.pad_token_id)
            return logits, loss
        
        return logits, None
"""
Transformer Seq2Seq implementation with Encoder-Decoder architecture.
Covers: embeddings, positional encoding, multi-head attention, feed-forward, and training.
"""
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

# Global constants for demos
V = 50000      # vocabulary size
d_model = 768  # model hidden dimension (embedding size)


class InputEmbeddings(nn.Module):
    """Maps token IDs (integer indices) to continuous d_model-dimensional vectors."""
    def __init__(self, d_model: int, vocab_size: int, device=None):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        # Lookup table: vocab_size rows, d_model cols. Input must be Long/Int indices.
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # x: (B, L) token ids -> output: (B, L, d_model)
        return self.embedding(x)


# Demo: batch of token IDs with shape (B, L)
B, L = 2, 6
token_ids = torch.randint(0, V, (B, L))


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding (Vaswani et al.).
    Adds position information to embeddings so the model knows token order.
    Uses sin for even indices, cos for odd indices; div_term = 10000^(-2i/d_model).
    """
    def __init__(self, d_model: int, seq_len: int, dropout: float, device=None) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        self.device = device or "cpu"

        pe = torch.zeros(seq_len, d_model, device=self.device)
        position = torch.arange(0, seq_len, dtype=torch.float, device=self.device).unsqueeze(1)  # (seq_len, 1)
        # div_term for scaling: 10000^(-2i/d_model) for i in 0,2,4,...
        div_term = torch.exp(torch.arange(0, d_model, 2, device=self.device).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, seq_len, d_model) for broadcasting over batch
        self.register_buffer('pe', pe)  # Not a parameter, but saved with model

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].requires_grad_(False)  # Add PE, no grad
        return self.dropout(x)


def causal_mask(L: int, device=None):
    """
    Creates causal (autoregressive) mask for decoder.
    Returns (L, L) bool tensor: True = positions to mask (future tokens).
    Upper triangle (excluding diagonal) is True so position i cannot attend to j > i.
    """
    device = device or "cpu"
    return torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)


class MultiHeadSelfAttention(nn.Module):
    """
    Scaled dot-product self-attention with multiple heads.
    Q, K, V all come from the same input x. Used in encoder and decoder self-attention.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, device=None):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        device = device or "cpu"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads  # dimension per head

        self.Wq = nn.Linear(d_model, d_model, bias=False, device=device)
        self.Wk = nn.Linear(d_model, d_model, bias=False, device=device)
        self.Wv = nn.Linear(d_model, d_model, bias=False, device=device)
        self.Wo = nn.Linear(d_model, d_model, bias=False, device=device)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        # x: (B, L, d_model)  attn_mask: (L, L) bool, True = masked
        B, L, _ = x.shape

        # Project to Q, K, V
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)

        # Split into heads: (B, L, d) -> (B, n_heads, L, d_head)
        q = q.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        # Scaled dot-product attention: scores = QK^T / sqrt(d_k)
        scores = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)  # (B, h, L, L)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values, then concat heads
        out = attn @ v  # (B, h, L, d_head)
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.Wo(out)


# --- Demo: multi-head attention (encoder vs decoder) ---
B, L, d_model, h = 2, 8, 64, 8
x = torch.randn(B, L, d_model)
mha = MultiHeadSelfAttention(d_model, h)

y_enc = mha(x)  # encoder self-attn (no causal mask)

y_enc

mask = causal_mask(L, device=x.device)
y_dec = mha(x, attn_mask=mask)  # decoder self-attn (causal)
print(y_enc.shape, y_dec.shape)



class FeedForward(nn.Module):
    """
    Position-wise feed-forward network: two linear layers with GELU in between.
    FFN(x) = W2 * GELU(W1 * x). Expands to d_ff then projects back to d_model.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0, device=None):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff, device=device)
        self.fc2 = nn.Linear(d_ff, d_model, device=device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)  # GELU often used instead of ReLU in transformers
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class EncoderLayer(nn.Module):
    """
    Single encoder block: self-attention + FFN, each with pre-norm and residual.
    No causal mask (encoder sees full sequence).
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1, device=None):
        super().__init__()
        self.device = device or "cpu"
        self.ln1 = nn.LayerNorm(d_model, device=self.device)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout=dropout, device=self.device)
        self.ln2 = nn.LayerNorm(d_model, device=self.device)
        self.ffn = FeedForward(d_model, d_ff, dropout=dropout, device=self.device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        # Pre-norm + residual
        x = x + self.dropout(self.attn(self.ln1(x), attn_mask=attn_mask))
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x


class Encoder(nn.Module):
    """
    Full encoder: stack of encoder layers. Expects pre-embedded input (B, L, d_model).
    Note: This encoder does NOT apply embedding in forward; parent handles token->embed.
    """
    def __init__(self, vocab_size: int, n_layers: int, d_model: int, n_heads: int, d_ff: int, max_seq_len: int, dropout: float = 0.1, device=None):
        super().__init__()
        self.device = device or "cpu"
        self.d_model = d_model
        self.num_layers = n_layers

        self.embedding = InputEmbeddings(d_model, vocab_size, device=self.device)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout, device=self.device)

        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout, device=self.device) for _ in range(n_layers)])
        self.ln_final = nn.LayerNorm(d_model, device=self.device)

    def forward(self, x, attn_mask=None):
        # x already embedded; pass through layers
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)
        return self.ln_final(x)


# --- Demo: encoder (expects embedded input, no embedding in forward) ---
B, L, d_model = 2, 10, 64
x0 = torch.randn(B, L, d_model)
enc = Encoder(vocab_size=V, n_layers=4, d_model=d_model, n_heads=8, d_ff=256, max_seq_len=L, dropout=0.1)
memory = enc(x0)
print(memory.shape)  # (B,L,d_model)


class MultiHeadAttention(nn.Module):
    """
    General multi-head attention: Q from x_q, K/V from x_kv (can differ).
    Used for: decoder self-attention (x_q=x_kv) and cross-attention (x_q=decoder, x_kv=encoder).
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, device=None):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.Wq = nn.Linear(d_model, d_model, bias=False, device=device)
        self.Wk = nn.Linear(d_model, d_model, bias=False, device=device)
        self.Wv = nn.Linear(d_model, d_model, bias=False, device=device)
        self.Wo = nn.Linear(d_model, d_model, bias=False, device=device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_q, x_kv, attn_mask=None):
        # x_q: (B, Lq, d) query sequence   x_kv: (B, Lk, d) key/value sequence
        B, Lq, _ = x_q.shape
        _, Lk, _ = x_kv.shape

        q = self.Wq(x_q).view(B, Lq, self.n_heads, self.d_head).transpose(1, 2)
        k = self.Wk(x_kv).view(B, Lk, self.n_heads, self.d_head).transpose(1, 2)
        v = self.Wv(x_kv).view(B, Lk, self.n_heads, self.d_head).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)  # (B, h, Lq, Lk)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v  # (B, h, Lq, d_head)
        out = out.transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        return self.Wo(out)


class DecoderLayer(nn.Module):
    """
    Single decoder block: masked self-attention (Q,K,V from decoder) + cross-attention (Q from decoder,
    K,V from encoder memory) + FFN. All with pre-norm and residual connections.
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1, device=None):
        super().__init__()
        self.device = device or "cpu"
        self.ln1 = nn.LayerNorm(d_model, device=self.device)
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout, device=self.device)
        self.ln2 = nn.LayerNorm(d_model, device=self.device)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout, device=self.device)
        self.ln3 = nn.LayerNorm(d_model, device=self.device)
        self.ffn = FeedForward(d_model, d_ff, dropout, device=self.device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, self_mask=None, cross_mask=None):
        # Masked self-attention: decoder attends to its own previous positions only
        x = x + self.dropout(self.self_attn(self.ln1(x), self.ln1(x), attn_mask=self_mask))
        # Cross-attention: decoder attends to encoder output (memory)
        x = x + self.dropout(self.cross_attn(self.ln2(x), memory, attn_mask=cross_mask))
        x = x + self.dropout(self.ffn(self.ln3(x)))
        return x


class Decoder(nn.Module):
    """
    Full decoder: embed + add positional encoding, then stack of decoder layers.
    Input x: (B, Lt) token IDs. Output: (B, Lt, d_model) contextualized representations.
    """
    def __init__(self, n_layers: int, d_model: int, n_heads: int, d_ff: int, vocab_size: int, max_seq_len: int, dropout: float = 0.1, device=None):
        super().__init__()
        self.device = device or "cpu"
        self.embedding = InputEmbeddings(d_model, vocab_size, device=self.device)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout, device=self.device)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout, device=self.device) for _ in range(n_layers)])
        self.ln_final = nn.LayerNorm(d_model, device=self.device)

    def forward(self, x, memory, self_mask=None, cross_mask=None):
        x = self.embedding(x)  # (B, Lt) -> (B, Lt, d_model)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, memory, self_mask=self_mask, cross_mask=cross_mask)
        return self.ln_final(x)


# -----------------------------------------------------------------------------
# Transformer Seq2Seq: full model combining encoder and decoder
# -----------------------------------------------------------------------------

class TransformerSeq2Seq(nn.Module):
    """
    Full transformer for sequence-to-sequence (e.g. translation).
    Source: embed + learnable pos -> encoder -> memory.
    Target: decoder (owns its embed + sinusoidal pos) attends to memory.
    Output: logits over target vocabulary via lm_head.
    """
    def __init__(self, V_src, V_tgt, d_model=256, n_heads=8, d_ff=1024, n_layers=4, L_max=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.src_tok = nn.Embedding(V_src, d_model)
        self.tgt_tok = nn.Embedding(V_tgt, d_model)
        self.pos_emb = nn.Embedding(L_max, d_model)  # learnable positional embeddings for source

        self.encoder = Encoder(vocab_size=V_src, n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_ff=d_ff, max_seq_len=L_max, dropout=dropout)
        self.decoder = Decoder(n_layers, d_model, n_heads, d_ff, vocab_size=V_tgt, max_seq_len=L_max, dropout=dropout)

        self.lm_head = nn.Linear(d_model, V_tgt, bias=False)  # project to vocab for next-token prediction

    def add_pos(self, x):
        """Add learnable positional embeddings. x: (B, L, d_model)."""
        B, L, _ = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)  # (B, L) position indices
        return x + self.pos_emb(pos)

    def forward(self, src_ids, tgt_in_ids):
        # src_ids: (B, Ls) source token IDs
        # tgt_in_ids: (B, Lt) target input (shifted-right, i.e. teacher forcing input)
        B, Ls = src_ids.shape
        _, Lt = tgt_in_ids.shape

        # Encode source
        src = self.add_pos(self.src_tok(src_ids))
        memory = self.encoder(src)  # (B, Ls, d_model)

        # Decode: pass token IDs; decoder does its own embedding + positional encoding
        self_mask = causal_mask(Lt, device=src_ids.device)  # prevent attending to future tokens
        dec_out = self.decoder(tgt_in_ids, memory, self_mask=self_mask)

        logits = self.lm_head(dec_out)  # (B, Lt, V_tgt)
        return logits


@torch.no_grad()
def greedy_decode(model, src_ids, bos_id, eos_id, max_len=64):
    """
    Autoregressive decoding: at each step pick the token with highest logit.
    Stops when all sequences in batch produce EOS. No gradients needed.
    """
    model.eval()
    B = src_ids.size(0)
    device = src_ids.device
    generated = torch.full((B, 1), bos_id, dtype=torch.long, device=device)  # start with <bos>

    for _ in range(max_len - 1):
        logits = model(src_ids, generated)      # (B, Lt, V)
        next_logits = logits[:, -1, :]          # (B, V) logits for next token
        next_id = torch.argmax(next_logits, dim=-1, keepdim=True)  # (B, 1) greedy pick
        generated = torch.cat([generated, next_id], dim=1)

        if torch.all(next_id.squeeze(1) == eos_id):
            break

    return generated


def training_step(model, optimizer, src_ids, tgt_ids, pad_id):
    """
    One training step: teacher forcing (tgt_in = tgt[:, :-1], target = tgt[:, 1:]).
    Cross-entropy over next-token prediction; pad tokens are ignored.
    """
    model.train()
    tgt_in  = tgt_ids[:, :-1]   # decoder input: [bos, t1, t2, ...]
    tgt_out = tgt_ids[:, 1:]    # target: [t1, t2, ..., eos]

    logits = model(src_ids, tgt_in)  # (B, T-1, V)
    V = logits.size(-1)

    # Flatten for cross_entropy; ignore padding
    loss = F.cross_entropy(
        logits.reshape(-1, V),
        tgt_out.reshape(-1),
        ignore_index=pad_id
    )

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return loss.item()


# =============================================================================
# Demo: build model, run one training step, and greedy decode
# =============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Vocabulary and special tokens
V_src, V_tgt = 10000, 12000
pad_id, bos_id, eos_id = 0, 1, 2

# Model and optimizer
model = TransformerSeq2Seq(V_src, V_tgt, d_model=128, n_heads=8, d_ff=512, n_layers=2).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

# Fake batch: random token IDs; first col = BOS, last col = EOS
B, Ls, Lt = 2, 12, 10
src = torch.randint(0, V_src, (B, Ls), device=device)
tgt = torch.randint(3, V_tgt, (B, Lt), device=device)
tgt[:, 0] = bos_id
tgt[:, -1] = eos_id

# Train one step
loss = training_step(model, opt, src, tgt, pad_id=pad_id)
print("loss:", loss)

# Greedy decode from same source
gen = greedy_decode(model, src, bos_id=bos_id, eos_id=eos_id, max_len=16)
print("generated:", gen)

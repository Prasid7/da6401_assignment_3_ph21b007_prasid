"""
model.py — Transformer Architecture Skeleton
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────┐
  │  scaled_dot_product_attention(Q, K, V, mask) → (out, weights)  │
  │  MultiHeadAttention.forward(q, k, v, mask)   → Tensor          │
  │  PositionalEncoding.forward(x)               → Tensor          │
  │  make_src_mask(src, pad_idx)                 → BoolTensor      │
  │  make_tgt_mask(tgt, pad_idx)                 → BoolTensor      │
  │  Transformer.encode(src, src_mask)           → Tensor          │
  │  Transformer.decode(memory,src_m,tgt,tgt_m)  → Tensor          │
  └─────────────────────────────────────────────────────────────────┘
"""

import math
import copy
import os
import gdown
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════
#   STANDALONE ATTENTION FUNCTION  
#    Exposed at module level so the autograder can import and test it
#    independently of MultiHeadAttention.
# ══════════════════════════════════════════════════════════════════════

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Scaled Dot-Product Attention.

        Attention(Q, K, V) = softmax( Q·Kᵀ / √dₖ ) · V

    Args:
        Q    : Query tensor,  shape (..., seq_q, d_k)
        K    : Key tensor,    shape (..., seq_k, d_k)
        V    : Value tensor,  shape (..., seq_k, d_v)
        mask : Optional Boolean mask, shape broadcastable to
               (..., seq_q, seq_k).
               Positions where mask is True are MASKED OUT
               (set to -inf before softmax).

    Returns:
        output : Attended output,   shape (..., seq_q, d_v)
        attn_w : Attention weights, shape (..., seq_q, seq_k)
    """
    
    # get the last dimension of Q for scaling
    d_k = Q.size(-1)

    # compute raw attention scores without scaling or masking
    raw_scores = torch.matmul(Q, K.transpose(-2, -1)) # shape (-----, seq_q, seq_k)

    # scale the scores by sqrt(d_k)
    scores = raw_scores / math.sqrt(d_k) 

    # apply mask if provided
    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))

    # compute attention weights
    attn_w = F.softmax(scores, dim=-1)

    # compute attended output
    output = torch.matmul(attn_w, V)

    return output, attn_w


# ══════════════════════════════════════════════════════════════════════
# ❷  MASK HELPERS 
#    Exposed at module level so they can be tested independently and
#    reused inside Transformer.forward.
# ══════════════════════════════════════════════════════════════════════

def make_src_mask(
    src: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """
    Build a padding mask for the encoder (source sequence).

    Args:
        src     : Source token-index tensor, shape [batch, src_len]
        pad_idx : Vocabulary index of the <pad> token (default 1)

    Returns:
        Boolean mask, shape [batch, 1, 1, src_len]
        True  → position is a PAD token (will be masked out)
        False → real token
    """
    # src shape: [batch, src_len]
    # we want to create a mask of shape [batch, 1, 1, src_len] where True indicates padding positions
    src_mask = (src == pad_idx).unsqueeze(1).unsqueeze(2) # shape [batch, 1, 1, src_len]
    return src_mask


def make_tgt_mask(
    tgt: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """
    Build a combined padding + causal (look-ahead) mask for the decoder.

    Args:
        tgt     : Target token-index tensor, shape [batch, tgt_len]
        pad_idx : Vocabulary index of the <pad> token (default 1)

    Returns:
        Boolean mask, shape [batch, 1, tgt_len, tgt_len]
        True → position is masked out (PAD or future token)
    """
    # tgt shape: [batch, tgt_len]
    # we want to create a mask of shape [batch, 1, tgt_len, tgt_len] where True indicates masked positions

    # first create padding mask
    tgt_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2) # shape [batch, 1, 1, tgt_len]

    # create causal mask (look-ahead mask)
    seq_len = tgt.size(-1)
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=tgt.device), diagonal=1)
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(1) # shape [1, 1, tgt_len, tgt_len]

    # combine padding and causal masks, True where either padding or future token
    combined_mask = tgt_mask | causal_mask # shape [batch, 1, tgt_len, tgt_len]

    return combined_mask


# ══════════════════════════════════════════════════════════════════════
#  MULTI-HEAD ATTENTION 
# ══════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention as in "Attention Is All You Need", §3.2.2.

        MultiHead(Q,K,V) = Concat(head_1,...,head_h) · W_O
        head_i = Attention(Q·W_Qi, K·W_Ki, V·W_Vi)

    You are NOT allowed to use torch.nn.MultiheadAttention.

    Args:
        d_model   (int)  : Total model dimensionality. Must be divisible by num_heads.
        num_heads (int)  : Number of parallel attention heads h.
        dropout   (float): Dropout probability applied to attention weights.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads   # depth per head
        
        self.dropout = nn.Dropout(p=dropout)

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
    
    def forward(
        self,
        query: torch.Tensor,
        key:   torch.Tensor,
        value: torch.Tensor,
        mask:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query : shape [batch, seq_q, d_model]
            key   : shape [batch, seq_k, d_model]
            value : shape [batch, seq_k, d_model]
            mask  : Optional BoolTensor broadcastable to
                    [batch, num_heads, seq_q, seq_k]
                    True → masked out (attend nowhere)

        Returns:
            output : shape [batch, seq_q, d_model]

        """
        batch_size = query.size(0)

        # Linear projections and split into heads
        # After projection: shape [batch, seq_len, d_model]
        # After split:      shape [batch, num_heads, seq_len, d_k]
        Q = self.W_Q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention for each head
        # attn_output shape: [batch, num_heads, seq_q, d_k]
        # attn_weights shape: [batch, num_heads, seq_q, seq_k]
        attn_output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)

        # apply dropout to attention output
        attn_output = self.dropout(attn_output)

        # Concatenate heads and apply final linear projection
        # Before concat: shape [batch, num_heads, seq_q, d_k]
        # After concat:  shape [batch, seq_q, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        output = self.W_O(attn_output)  # shape [batch, seq_q, d_model]

        return output


# ══════════════════════════════════════════════════════════════════════
#   POSITIONAL ENCODING  
# ══════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding as in "Attention Is All You Need", §3.5.

    Args:
        d_model  (int)  : Embedding dimensionality.
        dropout  (float): Dropout applied after adding encodings.
        max_len  (int)  : Maximum sequence length to pre-compute (default 5000).
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.max_len = max_len

        pe = torch.zeros(max_len, d_model) # positional encodings matrix, shape (max_len, d_model) to hold the 
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # hold position indices, shape (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # dimension indices (even positions) shape (d_model/2,)

        # apply the sinusoidal formula to fill the positional encoding matrix, shape (max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term) # apply sin to even dimensions
        pe[:, 1::2] = torch.cos(position * div_term) # apply cos to odd dimensions

        # add batch dimension
        pe = pe.unsqueeze(0) # shape (1, max_len, d_model)

        # register pe as a buffer so it's saved with the model but not trained
        self.register_buffer('pe', pe)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Input embeddings, shape [batch, seq_len, d_model]

        Returns:
            Tensor of same shape [batch, seq_len, d_model]
            = x  +  PE[:, :seq_len, :]  

        """

        seq_len = x.size(1) # get the sequence length from the input

        # add positional encoding to input embeddings, 
        # shape of pe[:, :seq_len, :] is (1, seq_len, d_model) and will broadcast across the batch dimension
        x = x + self.pe[:, :seq_len, :] 

        return self.dropout(x) # apply dropout after adding positional encoding
        


# ══════════════════════════════════════════════════════════════════════
#  FEED-FORWARD NETWORK 
# ══════════════════════════════════════════════════════════════════════

class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network, §3.3:

        FFN(x) = max(0, x·W₁ + b₁)·W₂ + b₂

    Args:
        d_model (int)  : Input / output dimensionality (e.g. 512).
        d_ff    (int)  : Inner-layer dimensionality (e.g. 2048).
        dropout (float): Dropout applied between the two linears.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)     # first linear layer projects from d_model to d_ff
        self.linear2 = nn.Linear(d_ff, d_model)     # second linear layer projects back from d_ff to d_model
        self.dropout = nn.Dropout(p=dropout)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : shape [batch, seq_len, d_model]
        Returns:
              shape [batch, seq_len, d_model]
        
        """
        x = self.linear1(x)         # shape [batch, seq_len, d_ff]
        x = torch.relu(x)           # apply ReLU non-linearity
        x = self.dropout(x)         # apply dropout after the first linear and activation
        x = self.linear2(x)         # shape [batch, seq_len, d_model] project back to d_model

        return x



# ══════════════════════════════════════════════════════════════════════
#  ENCODER LAYER  
# ══════════════════════════════════════════════════════════════════════

class EncoderLayer(nn.Module):
    """
    Single Transformer encoder sub-layer:
        x → [Self-Attention → Add & Norm] → [FFN → Add & Norm]

    Args:
        d_model   (int)  : Model dimensionality.
        num_heads (int)  : Number of attention heads.
        d_ff      (int)  : FFN inner dimensionality.
        dropout   (float): Dropout probability.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = dropout

        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)


    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]

        Returns:
            shape [batch, src_len, d_model]

        """

        # Self-Attention sub-layer
        attn_output = self.self_attn(x, x, x, src_mask)   # shape [batch, src_len, d_model]
        attn_output = self.dropout1(attn_output)          # Apply dropout to attention output
        x = self.norm1(x + attn_output)                   # residual connection + LayerNorm

        # Feed-Forward sub-layer
        ffn_output = self.ffn(x)                        # shape [batch, src_len, d_model]
        ffn_output = self.dropout2(ffn_output)          # Apply dropout to FFN output
        x = self.norm2(x + ffn_output)                  # residual connection + LayerNorm

        return x



# ══════════════════════════════════════════════════════════════════════
#   DECODER LAYER 
# ══════════════════════════════════════════════════════════════════════

class DecoderLayer(nn.Module):
    """
    Single Transformer decoder sub-layer:
        x → [Masked Self-Attn → Add & Norm]
          → [Cross-Attn(memory) → Add & Norm]
          → [FFN → Add & Norm]

    Args:
        d_model   (int)  : Model dimensionality.
        num_heads (int)  : Number of attention heads.
        d_ff      (int)  : FFN inner dimensionality.
        dropout   (float): Dropout probability.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = dropout

        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout) # masked self-attention for the decoder
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout) # cross-attention over the encoder output (memory)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout) # position-wise feed-forward network
        self.norm1 = nn.LayerNorm(d_model) # LayerNorm after masked self-attention
        self.norm2 = nn.LayerNorm(d_model) # LayerNorm after cross-attention
        self.norm3 = nn.LayerNorm(d_model) # LayerNorm after feed-forward network
        self.dropout1 = nn.Dropout(p=dropout) # Dropout after masked self-attention
        self.dropout2 = nn.Dropout(p=dropout) # Dropout after cross-attention
        self.dropout3 = nn.Dropout(p=dropout) # Dropout after feed-forward network


    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, tgt_len, d_model]
            memory   : Encoder output, shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            shape [batch, tgt_len, d_model]
        """
        
        # masked self-attention, query, key, value all come from the decoder input x, and
        # tgt_mask to mask out future tokens and padding
        self_attn_output = self.self_attn(x, x, x, tgt_mask) # shape [batch, tgt_len, d_model]
        self_attn_output = self.dropout1(self_attn_output) # apply dropout to masked self-attention output
        x = self.norm1(x + self_attn_output) # residual connection + LayerNorm

        # cross-attention where query comes from the decoder input x, and key/value come from the encoder output memory, and 
        # src_mask to mask out padding in the encoder output
        cross_attn_output = self.cross_attn(x, memory, memory, src_mask) # shape [batch, tgt_len, d_model]
        cross_attn_output = self.dropout2(cross_attn_output) # apply dropout to cross-attention output
        x = self.norm2(x + cross_attn_output) # residual connection + LayerNorm

        # feed-forward network
        ffn_output = self.ffn(x) # shape [batch, tgt_len, d_model]
        ffn_output = self.dropout3(ffn_output) # apply dropout to FFN output
        x = self.norm3(x + ffn_output) # residual connection + LayerNorm

        return x



# ══════════════════════════════════════════════════════════════════════
#  ENCODER & DECODER STACKS
# ══════════════════════════════════════════════════════════════════════

class Encoder(nn.Module):
    """Stack of N identical EncoderLayer modules with final LayerNorm."""

    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        
        self.N = N

        # create a stack of N identical encoder layers by deep copying the provided layer
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)]) 

        # final LayerNorm after the stack of encoder layers
        self.norm = nn.LayerNorm(layer.d_model) 


    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x    : shape [batch, src_len, d_model]
            mask : shape [batch, 1, 1, src_len]
        Returns:
            shape [batch, src_len, d_model]
        """
        
        for layer in self.layers:
            x = layer(x, mask) # pass the input through each encoder layer in the stack, shape remains [batch, src_len, d_model]

        return self.norm(x) # apply final LayerNorm to the output of the last encoder layer


class Decoder(nn.Module):
    """Stack of N identical DecoderLayer modules with final LayerNorm."""

    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()

        self.N = N

        # create a stack of N identical decoder layers by deep copying the provided layer
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)]) 

        # final LayerNorm after the stack of decoder layers
        self.norm = nn.LayerNorm(layer.d_model) 


    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, tgt_len, d_model]
            memory   : shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]
        Returns:
            shape [batch, tgt_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask) # pass the input through each decoder layer in the stack, shape remains [batch, tgt_len, d_model]

        return self.norm(x) # apply final LayerNorm to the output of the last decoder layer


# ══════════════════════════════════════════════════════════════════════
#   FULL TRANSFORMER  
# ══════════════════════════════════════════════════════════════════════

class Transformer(nn.Module):
    """
    Full Encoder-Decoder Transformer for sequence-to-sequence tasks.

    Args:
        src_vocab_size (int)  : Source vocabulary size.
        tgt_vocab_size (int)  : Target vocabulary size.
        d_model        (int)  : Model dimensionality (default 512).
        N              (int)  : Number of encoder/decoder layers (default 6).
        num_heads      (int)  : Number of attention heads (default 8).
        d_ff           (int)  : FFN inner dimensionality (default 2048).
        dropout        (float): Dropout probability (default 0.1).
    """

   
    def __init__(
        self,
        d_model:   int   = 512,
        N:         int   = 6,
        num_heads: int   = 8,
        d_ff:      int   = 2048,
        dropout:   float = 0.1,
    ) -> None:
        super().__init__()

        import spacy
        import subprocess
        import sys
        from datasets import load_dataset

        # load spacy tokenizers for German and English, downloading the models if not already available

        try:
            self.src_tokenizer = spacy.load("de_core_news_sm")
        except OSError:
            subprocess.run(
                [sys.executable, "-m", "spacy", "download", "de_core_news_sm"],
                check=True
            )
            self.src_tokenizer = spacy.load("de_core_news_sm")

        try:
            self.tgt_tokenizer = spacy.load("en_core_web_sm")
        except OSError:
            subprocess.run(
                [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                check=True
            )
            self.tgt_tokenizer = spacy.load("en_core_web_sm")

        # download the checkpoint containing the vocabularies and model weights, if available, and load it into memory

        checkpoint_path = "checkpoint_best.pth"
        checkpoint_state = None

        try:
            result = gdown.download(
                id="1xpB9nQxJVZ0feyZNqwYs6eD3vmrnGVrZ",
                output=checkpoint_path,
                quiet=False
            )

            if result is not None:
                checkpoint_state = torch.load(
                    checkpoint_path,
                    map_location="cpu"
                )
                print("Checkpoint downloaded successfully.")

        except Exception as e:
            print(f"Warning: Could not download checkpoint: {e}")

        # load vocabularies from checkpoint if available, otherwise build vocabularies from the training split of the Multi30k dataset using the tokenizers

        if (
            checkpoint_state is not None
            and isinstance(checkpoint_state, dict)
            and "src_vocab" in checkpoint_state
        ):

            self.src_vocab = checkpoint_state["src_vocab"]
            self.tgt_vocab = checkpoint_state["tgt_vocab"]

            print("Loaded vocab from checkpoint.")

        else:

            print("Building vocab from dataset...")

            special_tokens = ["<unk>", "<pad>", "<sos>", "<eos>"]

            train_data = load_dataset("bentrevett/multi30k")["train"]

            src_word2idx = {
                tok: i for i, tok in enumerate(special_tokens)
            }

            tgt_word2idx = {
                tok: i for i, tok in enumerate(special_tokens)
            }

            for entry in train_data:

                for token in self.src_tokenizer(entry["de"]):

                    if token.text not in src_word2idx:
                        src_word2idx[token.text] = len(src_word2idx)

                for token in self.tgt_tokenizer(entry["en"]):

                    if token.text not in tgt_word2idx:
                        tgt_word2idx[token.text] = len(tgt_word2idx)

            self.src_vocab = {
                "word2idx": src_word2idx,
                "idx2word": {
                    i: w for w, i in src_word2idx.items()
                },
                "vocab_size": len(src_word2idx),
                "pad_idx": src_word2idx["<pad>"],
                "unk_idx": src_word2idx["<unk>"],
                "sos_idx": src_word2idx["<sos>"],
                "eos_idx": src_word2idx["<eos>"],
            }

            self.tgt_vocab = {
                "word2idx": tgt_word2idx,
                "idx2word": {
                    i: w for w, i in tgt_word2idx.items()
                },
                "vocab_size": len(tgt_word2idx),
                "pad_idx": tgt_word2idx["<pad>"],
                "unk_idx": tgt_word2idx["<unk>"],
                "sos_idx": tgt_word2idx["<sos>"],
                "eos_idx": tgt_word2idx["<eos>"],
            }

        # store the padding, start-of-sequence, and end-of-sequence token indices for easy access when creating masks and during inference

        self.src_pad_idx = self.src_vocab["pad_idx"]
        self.tgt_pad_idx = self.tgt_vocab["pad_idx"]

        self.tgt_sos_idx = self.tgt_vocab["sos_idx"]
        self.tgt_eos_idx = self.tgt_vocab["eos_idx"]

        # vocab sizes for the source and target languages, needed for embedding layers and output projection layer

        src_vocab_size = self.src_vocab["vocab_size"]
        tgt_vocab_size = self.tgt_vocab["vocab_size"]

        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size

        # store the model hyperparameters for use when building the model and in the forward pass

        self.d_model = d_model
        self.N = N
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = dropout

        # build the model components: source and target token embeddings, positional encoding, encoder and decoder stacks, and final output projection layer

        self.src_embedding = nn.Embedding(
            src_vocab_size,
            d_model
        )

        self.tgt_embedding = nn.Embedding(
            tgt_vocab_size,
            d_model
        )

        self.positional_encoding = PositionalEncoding(
            d_model,
            dropout
        )

        encoder_layer = EncoderLayer(
            d_model,
            num_heads,
            d_ff,
            dropout
        )

        self.encoder = Encoder(
            encoder_layer,
            N
        )

        decoder_layer = DecoderLayer(
            d_model,
            num_heads,
            d_ff,
            dropout
        )

        self.decoder = Decoder(
            decoder_layer,
            N
        )

        self.output_projection = nn.Linear(
            d_model,
            tgt_vocab_size
        )

        # load model weights from checkpoint if available, otherwise the model will be randomly initialized as per PyTorch defaults

        if checkpoint_state is not None:

            try:

                if (
                    isinstance(checkpoint_state, dict)
                    and "model_state_dict" in checkpoint_state
                ):

                    self.load_state_dict(
                        checkpoint_state["model_state_dict"]
                    )

                else:
                    self.load_state_dict(checkpoint_state)

                print("Loaded model weights successfully.")

            except Exception as e:
                print(f"Warning: Could not load weights: {e}")



    # ── AUTOGRADER HOOKS ── keep these signatures exactly ─────────────

    def encode(
        self,
        src:      torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the full encoder stack.

        Args:
            src      : Token indices, shape [batch, src_len]
            src_mask : shape [batch, 1, 1, src_len]

        Returns:
            memory : Encoder output, shape [batch, src_len, d_model]
        """

        # shape [batch, src_len, d_model], scale the embeddings by sqrt(d_model) as in the paper
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model) 

        # add positional encoding to the source embeddings, shape remains [batch, src_len, d_model]
        src_emb = self.positional_encoding(src_emb) 

        # pass the embedded source through the encoder stack, output shape [batch, src_len, d_model]
        memory = self.encoder(src_emb, src_mask) 

        return memory 
    

    def decode(
        self,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt:      torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the full decoder stack and project to vocabulary logits.

        Args:
            memory   : Encoder output,  shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt      : Token indices,   shape [batch, tgt_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            logits : shape [batch, tgt_len, tgt_vocab_size]
        """

        # shape [batch, tgt_len, d_model], scale the embeddings by sqrt(d_model) as in the paper
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)

        # add positional encoding to the target embeddings, shape remains [batch, tgt_len, d_model]
        tgt_emb = self.positional_encoding(tgt_emb) 

         # pass the embedded target and encoder output through the decoder stack, output shape [batch, tgt_len, d_model]
        decoder_output = self.decoder(tgt_emb, memory, src_mask, tgt_mask)

        # project the decoder output to target vocabulary logits, shape [batch, tgt_len, tgt_vocab_size]    
        logits = self.output_projection(decoder_output) 

        return logits


    def forward(
        self,
        src:      torch.Tensor,
        tgt:      torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Full encoder-decoder forward pass.

        Args:
            src      : shape [batch, src_len]
            tgt      : shape [batch, tgt_len]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            logits : shape [batch, tgt_len, tgt_vocab_size]
        """

        # encode the source sequence to get the memory representation, shape [batch, src_len, d_model]
        memory = self.encode(src, src_mask)

        # decode the target sequence using the memory and masks to get the output logits, shape [batch, tgt_len, tgt_vocab_size]
        logits = self.decode(memory, src_mask, tgt, tgt_mask)

        return logits


    def infer(self, src_sentence: str) -> str:
        """
        Translates a German sentence to English using greedy autoregressive decoding.
        
        Args:
            src_sentence: The raw German text.
            
            
        Returns:
            The fully translated English string, detokenized and clean.
        """
        
        self.eval() # set model to eval mode

        with torch.inference_mode(): # disable gradient computation and autograd tracking for faster inference
            
            # sanity check to ensure vocabularies and tokenizers are available, return empty string if not
            if not hasattr(self, 'src_vocab') or self.src_vocab is None:
                return ""
            if not hasattr(self, 'tgt_vocab') or self.tgt_vocab is None:
                return ""

            # get the device of the model parameters to ensure tensors are on the same device
            device = next(self.parameters()).device 
            
            # tokenize input sentence 
            # if a tokenizer is provided, use it to tokenize the input sentence, otherwise fall back to simple whitespace splitting
            if hasattr(self, 'src_tokenizer') and self.src_tokenizer is not None: 
                src_tokens = [token.text for token in self.src_tokenizer(src_sentence)] 
            else: 
                src_tokens = src_sentence.strip().split()

            # add special tokens 
            src_tokens = ['<sos>'] + src_tokens + ['<eos>']

            # convert tokens to ids
            word2idx = self.src_vocab['word2idx']
            unk_idx = self.src_vocab['word2idx']['<unk>']
            src_ids = [word2idx.get(token, unk_idx) for token in src_tokens] # convert tokens to indices, use <unk> for unknown tokens

            # convert src_ids to a tensor and add batch dimension, shape [1, src_len]
            src_tensor = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0) 

            # create source mask for the input sentence, shape [1, 1, 1, src_len]
            src_mask = make_src_mask(src_tensor, self.src_pad_idx).to(device)

            # run encoder once on the source tensor to get the memory representation, shape [1, src_len, d_model]
            memory = self.encode(src_tensor, src_mask)

            # initalize the decoder input
            ys = torch.tensor([[self.tgt_sos_idx]], dtype=torch.long, device=device) # shape [1, 1], start with <sos>

            max_len = 50
            # autoregressive decoding loop
            for i in range(max_len): # generate up to max_len tokens (including <sos>)

                # create target mask for the current decoder input, shape [1, 1, out_len, out_len]
                tgt_mask = make_tgt_mask(ys, self.tgt_pad_idx).to(device)

                # run decoder on current ys; we only need the last position's logit
                # shape [1, out_len, tgt_vocab_size] -> take only last timestep
                logits = self.decode(memory, src_mask, ys, tgt_mask)
                next_token_logits = logits[:, -1, :]  # shape [1, tgt_vocab_size]

                # choose the token with the highest probability as the next token
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True) # shape [1, 1]

                # append the next token to the generated sequence
                ys = torch.cat([ys, next_token], dim=1) # shape [1, out_len]

                # stop if <eos> is generated
                if next_token.item() == self.tgt_eos_idx:
                    break
            
            # convert generated token indices back to tokens
            generated_indices = ys.squeeze(0).tolist() # shape [out_len], convert to

            idx2word = self.tgt_vocab['idx2word']
            generated_tokens = [idx2word.get(idx, '<unk>') for idx in generated_indices] # convert indices back to tokens, use <unk> for unknown indices

            # remove special tokens from the generated token list
            generated_tokens = [token for token in generated_tokens if token not in ['<sos>', '<eos>', '<pad>']]

            # detokenize: join with spaces, then re-attach punctuation
            # spacy splits "bench." into ["bench", "."], so naive join gives "bench ."
            # Sacrebleu penalises this — reattach punctuation for a fair score
            import re
            translated_sentence = " ".join(generated_tokens).strip()
            translated_sentence = re.sub(r" ([.,!?;:])", r"\1", translated_sentence)
            translated_sentence = re.sub(r" ('s|n't|'re|'ve|'ll|'d|'m)\b", r"\1", translated_sentence)

            return translated_sentence
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re
import math
import random
import os
from typing import Dict, List, Tuple, Optional
import PyPDF2
import io
from collections import Counter, defaultdict

class CustomTokenizer:
    """
    Custom tokenizer implementing BPE-like tokenization
    Converts text into sequences of integers
    """

    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.inverse_vocab = {}
        self.special_tokens = {
            '[PAD]': 0,
            '[UNK]': 1,
            '[CLS]': 2,
            '[SEP]': 3,
            '[MASK]': 4
        }
        self.vocab.update(self.special_tokens)
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.next_id = len(self.special_tokens)

    def build_vocab(self, texts: List[str]):
        """Build vocabulary from training texts"""
        word_counts = Counter()
        for text in texts:
            words = self._preprocess_text(text)
            word_counts.update(words)

        for word, count in word_counts.most_common(self.vocab_size - len(self.special_tokens)):
            if word not in self.vocab:
                self.vocab[word] = self.next_id
                self.inverse_vocab[self.next_id] = word
                self.next_id += 1

    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocess text into tokens"""
        text = text.lower()
        words = re.findall(r'\b\w+\b|[^\w\s]', text)
        return words

    def encode(self, text: str, max_length: Optional[int] = None, add_special_tokens: bool = True) -> List[int]:
        """Convert text to token IDs"""
        words = self._preprocess_text(text)

        if add_special_tokens:
            token_ids = [self.special_tokens['[CLS]']]
        else:
            token_ids = []

        for word in words:
            if word in self.vocab:
                token_ids.append(self.vocab[word])
            else:
                token_ids.append(self.special_tokens['[UNK]'])

        if add_special_tokens:
            token_ids.append(self.special_tokens['[SEP]'])

        if max_length:
            if len(token_ids) > max_length:
                token_ids = token_ids[:max_length-1] + [self.special_tokens['[SEP]']]
            else:
                token_ids.extend([self.special_tokens['[PAD]']] * (max_length - len(token_ids)))

        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """Convert token IDs back to text"""
        words = []
        for token_id in token_ids:
            if token_id in self.inverse_vocab:
                word = self.inverse_vocab[token_id]
                if word not in ['[PAD]', '[CLS]', '[SEP]']:
                    words.append(word)
        return ' '.join(words)

class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer
    Adds position information to embeddings
    """

    def __init__(self, d_model: int, max_seq_length: int = 512):
        super().__init__()
        self.d_model = d_model

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length).unsqueeze(1).float()

        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)

        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Add positional encoding to input embeddings"""
        seq_length = x.size(1)
        return x + self.pe[:, :seq_length, :]

class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism
    Core component of transformer encoder
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, x, mask=None):
        """
        Forward pass of multi-head attention
        x: (batch_size, seq_length, d_model)
        """
        batch_size, seq_length, _ = x.size()

        # Generate Q, K, V matrices
        Q = self.W_q(x)  # (batch_size, seq_length, d_model)
        K = self.W_k(x)  # (batch_size, seq_length, d_model)
        V = self.W_v(x)  # (batch_size, seq_length, d_model)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context = torch.matmul(attention_weights, V)

        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_length, self.d_model)

        output = self.W_o(context)

        return output, attention_weights

class FeedForward(nn.Module):
    """
    Feed-forward network for transformer
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerEncoderLayer(nn.Module):
    """
    Single transformer encoder layer with self-attention and feed-forward
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):

        attn_output, attention_weights = self.self_attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x, attention_weights

class TransformerEncoder(nn.Module):
    """
    Stack of transformer encoder layers
    """

    def __init__(self, vocab_size: int, d_model: int = 768, num_heads: int = 12,
                 num_layers: int = 6, d_ff: int = 3072, max_seq_length: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model

        self.token_embedding = nn.Embedding(vocab_size, d_model)

        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

        nn.init.normal_(self.token_embedding.weight, mean=0, std=0.02)

    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass through transformer encoder
        input_ids: (batch_size, seq_length)
        """
        embeddings = self.token_embedding(input_ids) * math.sqrt(self.d_model)

        x = self.positional_encoding(embeddings)
        x = self.dropout(x)

        all_attention_weights = []
        for layer in self.layers:
            x, attention_weights = layer(x, attention_mask)
            all_attention_weights.append(attention_weights)

        return x, all_attention_weights

class MLMTaskHead(nn.Module):
    """
    Masked Language Modeling task head for BERT-like prediction
    Predicts masked tokens based on context
    """

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.decoder = nn.Linear(d_model, vocab_size)

    def forward(self, hidden_states):
        """
        Forward pass for MLM prediction
        hidden_states: (batch_size, seq_length, d_model)
        Returns: (batch_size, seq_length, vocab_size)
        """

        x = self.dense(hidden_states)
        x = F.gelu(x)
        x = self.layer_norm(x)

        logits = self.decoder(x)

        return logits


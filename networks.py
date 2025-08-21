import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleHeadAttention(nn.Module): 
    """Implements single-head self-attention mechanism.
    
    Takes input sequence and projects it into Query, Key and Value vectors,
    then computes scaled dot-product attention.
    """
    def __init__(self, input_dim, query_dim, value_dim): 
        super().__init__()
        self.input_dim, self.query_dim, self.value_dim = input_dim, query_dim, value_dim
        # Linear projections for Q, K, V
        self.W_Q = nn.Linear(input_dim, query_dim, bias=False)
        self.W_K = nn.Linear(input_dim, query_dim, bias=False)
        self.W_V = nn.Linear(input_dim, value_dim, bias=False)

    def forward(self, x): 
        # x : (batch_size, seq_len, input_dim)

        # Project input into Q, K, V vectors
        Q = self.W_Q(x) # (batch_size, seq_len, query_dim)
        K = self.W_K(x) # (batch_size, seq_len, query_dim)
        V = self.W_V(x) # (batch_size, seq_len, value_dim)

        # Compute scaled dot-product attention
        scores = Q @ K.transpose(-2, -1) / (self.query_dim ** 0.5) # (batch_size, seq_len, seq_len)
        attention_weights = F.softmax(scores, dim=-1) # (batch_size, seq_len, seq_len)
        out = attention_weights @ V # (batch_size, seq_len, value_dim)

        return out


class MultiHeadAttention(nn.Module): 
    """Implements multi-head attention by running multiple attention heads in parallel
    and concatenating their outputs.
    """
    def __init__(self, input_dim, query_dim, value_dim, num_heads): 
        super().__init__()
        self.num_heads = num_heads
        # Create list of attention heads
        self.heads = nn.ModuleList([SingleHeadAttention(input_dim, query_dim, value_dim) for _ in range(self.num_heads)])
        # Project concatenated outputs back to input dimension
        self.W_out = nn.Linear(value_dim * num_heads, input_dim)

    def forward(self, x): 
        # Run all heads in parallel and concatenate outputs
        out_heads = torch.cat([head(x) for head in self.heads], dim=-1) # (batch_size, seq_len, value_dim * num_heads)
        out = self.W_out(out_heads) # (batch_size, seq_len, input_dim)
        return out

class FeedForward(nn.Module): 
    """Simple feed-forward network with one hidden layer and ReLU activation."""
    def __init__(self, input_dim, hidden_dim, output_dim): 
        super().__init__()
        self.W_1 = nn.Linear(input_dim, hidden_dim)
        self.W_2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x): 
        # Apply first layer with ReLU activation
        out = F.relu(self.W_1(x))
        # Project back to output dimension
        out = self.W_2(out)
        return out

class TransformerBlock(nn.Module): 
    """Basic Transformer block consisting of multi-head self-attention followed by
    a feed-forward network. Uses residual connections and layer normalization.
    """
    def __init__(self, input_dim, hidden_dim, num_heads): 
        super().__init__()
        self.attention = MultiHeadAttention(input_dim, input_dim, input_dim, num_heads)
        self.layernorm1 = nn.LayerNorm(input_dim)
        self.layernorm2 = nn.LayerNorm(input_dim)
        self.feed_forward = FeedForward(input_dim, hidden_dim, input_dim)

    def forward(self, x): 
        # Multi-head self-attention with residual connection and layer norm
        out = self.attention(x)
        out = self.layernorm1(out + x)
        # Feed-forward with residual connection and layer norm
        out = self.feed_forward(out)
        out = self.layernorm2(out + x)
        return out

class Transformer(nn.Module): 
    def __init__(self, vocab_size, embedding_size, num_heads, num_layers, block_size, dropout=0.1): 
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_size)
        self.position_embedding = nn.Embedding(block_size, embedding_size)
        self.blocks = nn.Sequential(*[TransformerBlock(embedding_size, embedding_size, num_heads) for _ in range(num_layers)])
        self.block_size = block_size

        # project back to the vocab size
        self.lm_head = nn.Linear(embedding_size, vocab_size)

    def forward(self, idx): 
        B, T = idx.shape
        token_embeddings = self.token_embedding(idx)
        position_embeddings = self.position_embedding(torch.arange(T, device=idx.device))
        x = token_embeddings + position_embeddings
        x = self.blocks(x)
        return self.lm_head(x)
"""
Attention module for the Jadio LLM.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Union


class JadioAttention(nn.Module):
    """
    Multi-head self-attention module for Jadio.
    
    Implements the scaled dot-product attention mechanism used in GPT-2,
    with causal masking for autoregressive generation.
    """
    
    def __init__(self,
                 hidden_size: int,
                 num_attention_heads: int,
                 attention_dropout: float = 0.1,
                 resid_dropout: float = 0.1,
                 max_position_embeddings: int = 1024,
                 scale_attn_weights: bool = True,
                 bias: bool = True):
        """
        Initialize the attention module.
        
        Args:
            hidden_size: Hidden dimension size
            num_attention_heads: Number of attention heads
            attention_dropout: Dropout probability for attention weights
            resid_dropout: Dropout probability for residual connection
            max_position_embeddings: Maximum sequence length for causal mask
            scale_attn_weights: Whether to scale attention weights
            bias: Whether to use bias in linear layers
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.scale_attn_weights = scale_attn_weights
        
        if hidden_size % num_attention_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_attention_heads ({num_attention_heads})")
        
        # Combined linear layer for q, k, v
        self.c_attn = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        
        # Output projection
        self.c_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        
        # Dropout layers
        self.attn_dropout = nn.Dropout(attention_dropout)
        self.resid_dropout = nn.Dropout(resid_dropout)
        
        # Causal mask
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_position_embeddings, max_position_embeddings, dtype=torch.bool)).view(
                1, 1, max_position_embeddings, max_position_embeddings
            ),
            persistent=False,
        )
        
        # Scale factor for attention scores
        self.scale = 1.0 / math.sqrt(self.head_dim) if scale_attn_weights else 1.0
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize the weights."""
        # Initialize with normal distribution
        nn.init.normal_(self.c_attn.weight, std=0.02)
        nn.init.normal_(self.c_proj.weight, std=0.02)
        
        # Initialize biases to zero
        if self.c_attn.bias is not None:
            nn.init.zeros_(self.c_attn.bias)
        if self.c_proj.bias is not None:
            nn.init.zeros_(self.c_proj.bias)
    
    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Split the last dimension into (num_heads, head_dim).
        
        Args:
            tensor: Input tensor of shape (batch_size, seq_len, hidden_size)
            
        Returns:
            Tensor of shape (batch_size, num_heads, seq_len, head_dim)
        """
        batch_size, seq_len, hidden_size = tensor.shape
        tensor = tensor.view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        return tensor.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, head_dim)
    
    def _merge_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Merge the head dimension back into the hidden dimension.
        
        Args:
            tensor: Input tensor of shape (batch_size, num_heads, seq_len, head_dim)
            
        Returns:
            Tensor of shape (batch_size, seq_len, hidden_size)
        """
        batch_size, num_heads, seq_len, head_dim = tensor.shape
        tensor = tensor.permute(0, 2, 1, 3)  # (batch_size, seq_len, num_heads, head_dim)
        return tensor.contiguous().view(batch_size, seq_len, self.hidden_size)
    
    def _attn(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
              attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply attention mechanism.
        
        Args:
            query: Query tensor of shape (batch_size, num_heads, seq_len, head_dim)
            key: Key tensor of shape (batch_size, num_heads, seq_len, head_dim)
            value: Value tensor of shape (batch_size, num_heads, seq_len, head_dim)
            attention_mask: Optional attention mask
            
        Returns:
            Tuple of (attended values, attention weights)
        """
        # Compute attention scores
        attn_scores = torch.matmul(query, key.transpose(-1, -2))  # (batch_size, num_heads, seq_len, seq_len)
        
        # Scale attention scores
        attn_scores = attn_scores * self.scale
        
        # Apply causal mask
        seq_len = query.size(-2)
        causal_mask = self.causal_mask[:, :, :seq_len, :seq_len]
        attn_scores = torch.where(causal_mask, attn_scores, torch.finfo(attn_scores.dtype).min)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        
        # Apply softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Apply dropout to attention weights
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value)
        
        return attn_output, attn_weights
    
    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                use_cache: bool = False,
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                output_attentions: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Forward pass of the attention module.
        
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)
            attention_mask: Optional attention mask
            use_cache: Whether to return key/value for caching
            past_key_value: Cached key/value from previous steps
            output_attentions: Whether to return attention weights
            
        Returns:
            Attention output or tuple with additional outputs
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Apply linear transformation to get q, k, v
        qkv = self.c_attn(hidden_states)
        query, key, value = qkv.split(self.hidden_size, dim=-1)
        
        # Split into heads
        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)
        
        # Use cached key/value if provided
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key = torch.cat([past_key, key], dim=-2)
            value = torch.cat([past_value, value], dim=-2)
        
        # Apply attention
        attn_output, attn_weights = self._attn(query, key, value, attention_mask)
        
        # Merge heads
        attn_output = self._merge_heads(attn_output)
        
        # Apply output projection
        attn_output = self.c_proj(attn_output)
        
        # Apply residual dropout
        attn_output = self.resid_dropout(attn_output)
        
        # Prepare outputs
        outputs = (attn_output,)
        
        if use_cache:
            outputs += ((key, value),)
        
        if output_attentions:
            outputs += (attn_weights,)
        
        return outputs[0] if len(outputs) == 1 else outputs


class JadioSelfAttention(nn.Module):
    """
    Complete self-attention block with residual connection and layer norm.
    
    This combines the attention mechanism with pre-layer normalization and
    residual connection, following the GPT-2 architecture.
    """
    
    def __init__(self,
                 hidden_size: int,
                 num_attention_heads: int,
                 attention_dropout: float = 0.1,
                 resid_dropout: float = 0.1,
                 max_position_embeddings: int = 1024,
                 scale_attn_weights: bool = True,
                 layer_norm_eps: float = 1e-5,
                 bias: bool = True):
        """
        Initialize the self-attention block.
        
        Args:
            hidden_size: Hidden dimension size
            num_attention_heads: Number of attention heads
            attention_dropout: Dropout probability for attention weights
            resid_dropout: Dropout probability for residual connection
            max_position_embeddings: Maximum sequence length
            scale_attn_weights: Whether to scale attention weights
            layer_norm_eps: Layer norm epsilon
            bias: Whether to use bias in linear layers
        """
        super().__init__()
        
        # Import here to avoid circular imports
        from .jadio_layer_norm import JadioLayerNorm
        
        # Layer normalization (pre-norm)
        self.ln = JadioLayerNorm(hidden_size, eps=layer_norm_eps)
        
        # Attention mechanism
        self.attn = JadioAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_dropout,
            resid_dropout=resid_dropout,
            max_position_embeddings=max_position_embeddings,
            scale_attn_weights=scale_attn_weights,
            bias=bias
        )
    
    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                use_cache: bool = False,
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                output_attentions: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Forward pass with residual connection.
        
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)
            attention_mask: Optional attention mask
            use_cache: Whether to return key/value for caching
            past_key_value: Cached key/value from previous steps
            output_attentions: Whether to return attention weights
            
        Returns:
            Attention output or tuple with additional outputs
        """
        # Pre-layer norm
        residual = hidden_states
        hidden_states = self.ln(hidden_states)
        
        # Attention
        attn_outputs = self.attn(
            hidden_states,
            attention_mask=attention_mask,
            use_cache=use_cache,
            past_key_value=past_key_value,
            output_attentions=output_attentions
        )
        
        if isinstance(attn_outputs, tuple):
            attn_output = attn_outputs[0]
            additional_outputs = attn_outputs[1:]
        else:
            attn_output = attn_outputs
            additional_outputs = ()
        
        # Residual connection
        output = residual + attn_output
        
        # Return with additional outputs if any
        if additional_outputs:
            return (output,) + additional_outputs
        else:
            return output


def main():
    """Test the attention module."""
    print("[Jadio] Testing Attention module...")
    
    # Test parameters
    batch_size = 2
    seq_len = 10
    hidden_size = 768
    num_heads = 12
    
    # Test basic attention
    print("\n--- Testing JadioAttention ---")
    attention = JadioAttention(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        attention_dropout=0.1,
        resid_dropout=0.1,
        max_position_embeddings=1024,
        scale_attn_weights=True
    )
    
    # Create test input
    x = torch.randn(batch_size, seq_len, hidden_size)
    print(f"Input shape: {x.shape}")
    print(f"Input mean: {x.mean():.4f}, std: {x.std():.4f}")
    
    # Forward pass
    out = attention(x)
    print(f"Output shape: {out.shape}")
    print(f"Output mean: {out.mean():.4f}, std: {out.std():.4f}")
    
    # Test with output attentions
    print("\n--- Testing with attention weights ---")
    out_with_attn = attention(x, output_attentions=True)
    attn_output, attn_weights = out_with_attn
    print(f"Attention output shape: {attn_output.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    print(f"Attention weights sum (should be ~1): {attn_weights.sum(dim=-1).mean():.4f}")
    
    # Test with caching
    print("\n--- Testing with caching ---")
    out_with_cache = attention(x, use_cache=True)
    attn_output, past_kv = out_with_cache
    print(f"Cached key shape: {past_kv[0].shape}")
    print(f"Cached value shape: {past_kv[1].shape}")
    
    # Test incremental generation
    next_token = torch.randn(batch_size, 1, hidden_size)
    out_incremental = attention(next_token, past_key_value=past_kv, use_cache=True)
    print(f"Incremental output shape: {out_incremental[0].shape}")
    
    # Test complete self-attention block
    print("\n--- Testing JadioSelfAttention ---")
    self_attn = JadioSelfAttention(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        attention_dropout=0.1,
        resid_dropout=0.1,
        max_position_embeddings=1024,
        scale_attn_weights=True
    )
    
    out_self = self_attn(x)
    print(f"Self-attention output shape: {out_self.shape}")
    print(f"Self-attention output mean: {out_self.mean():.4f}, std: {out_self.std():.4f}")
    
    # Check residual connection is working
    diff = (out_self - x).abs().mean()
    print(f"Difference from input (residual effect): {diff:.4f}")
    
    # Test causal masking
    print("\n--- Testing causal masking ---")
    # Create a simple pattern to see if causal masking works
    test_input = torch.zeros(1, 3, hidden_size)
    test_input[0, 0] = 1.0  # First position has signal
    test_input[0, 1] = 2.0  # Second position has different signal
    test_input[0, 2] = 3.0  # Third position has different signal
    
    attn_test = JadioAttention(hidden_size, num_heads, attention_dropout=0.0)
    with torch.no_grad():
        test_out = attn_test(test_input, output_attentions=True)
        test_attn_weights = test_out[1]
        
        print("Attention weights (should show causal pattern):")
        print("Position 0 attends to:", test_attn_weights[0, 0, 0].round(decimals=3))
        print("Position 1 attends to:", test_attn_weights[0, 0, 1].round(decimals=3))
        print("Position 2 attends to:", test_attn_weights[0, 0, 2].round(decimals=3))
    
    print("\n[Jadio] Attention test completed!")
"""
Feed forward module for the Jadio LLM.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


def gelu_new(x: torch.Tensor) -> torch.Tensor:
    """
    GELU activation function as used in GPT-2.
    
    This is the "new" GELU implementation that's more accurate than
    the original approximation.
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


def gelu_pytorch_tanh(x: torch.Tensor) -> torch.Tensor:
    """
    GELU activation using PyTorch's tanh approximation.
    """
    return F.gelu(x, approximate='tanh')


class JadioMLP(nn.Module):
    """
    Multi-layer perceptron (feed-forward) module for Jadio.
    
    This implements the standard transformer feed-forward network:
    - Linear projection to intermediate size (typically 4x hidden size)
    - Activation function (GELU)
    - Linear projection back to hidden size
    - Dropout
    """
    
    def __init__(self,
                 hidden_size: int,
                 intermediate_size: Optional[int] = None,
                 activation_function: str = "gelu_new",
                 dropout_prob: float = 0.1,
                 bias: bool = True):
        """
        Initialize the MLP.
        
        Args:
            hidden_size: Hidden dimension size
            intermediate_size: Intermediate dimension size (4 * hidden_size if None)
            activation_function: Activation function to use
            dropout_prob: Dropout probability
            bias: Whether to use bias in linear layers
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size or 4 * hidden_size
        self.activation_function = activation_function
        
        # First linear layer (hidden -> intermediate)
        self.c_fc = nn.Linear(hidden_size, self.intermediate_size, bias=bias)
        
        # Second linear layer (intermediate -> hidden)
        self.c_proj = nn.Linear(self.intermediate_size, hidden_size, bias=bias)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_prob)
        
        # Set activation function
        self.act = self._get_activation_function(activation_function)
        
        # Initialize weights
        self._init_weights()
    
    def _get_activation_function(self, activation_function: str):
        """Get the activation function."""
        if activation_function == "gelu":
            return F.gelu
        elif activation_function == "gelu_new":
            return gelu_new
        elif activation_function == "gelu_pytorch_tanh":
            return gelu_pytorch_tanh
        elif activation_function == "relu":
            return F.relu
        elif activation_function == "silu" or activation_function == "swish":
            return F.silu
        else:
            raise ValueError(f"Unsupported activation function: {activation_function}")
    
    def _init_weights(self):
        """Initialize the weights."""
        # Initialize with normal distribution
        nn.init.normal_(self.c_fc.weight, std=0.02)
        nn.init.normal_(self.c_proj.weight, std=0.02)
        
        # Initialize biases to zero
        if self.c_fc.bias is not None:
            nn.init.zeros_(self.c_fc.bias)
        if self.c_proj.bias is not None:
            nn.init.zeros_(self.c_proj.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size)
        """
        # First linear transformation
        x = self.c_fc(x)
        
        # Activation function
        x = self.act(x)
        
        # Second linear transformation
        x = self.c_proj(x)
        
        # Dropout
        x = self.dropout(x)
        
        return x


class JadioFeedForward(nn.Module):
    """
    Complete feed-forward block with residual connection and layer norm.
    
    This combines the MLP with pre-layer normalization and residual connection,
    following the GPT-2 architecture.
    """
    
    def __init__(self,
                 hidden_size: int,
                 intermediate_size: Optional[int] = None,
                 activation_function: str = "gelu_new",
                 dropout_prob: float = 0.1,
                 layer_norm_eps: float = 1e-5,
                 bias: bool = True):
        """
        Initialize the feed-forward block.
        
        Args:
            hidden_size: Hidden dimension size
            intermediate_size: Intermediate dimension size
            activation_function: Activation function to use
            dropout_prob: Dropout probability
            layer_norm_eps: Layer norm epsilon
            bias: Whether to use bias in linear layers
        """
        super().__init__()
        
        # Import here to avoid circular imports
        from .jadio_layer_norm import JadioLayerNorm
        
        # Layer normalization (pre-norm)
        self.ln = JadioLayerNorm(hidden_size, eps=layer_norm_eps)
        
        # MLP
        self.mlp = JadioMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation_function=activation_function,
            dropout_prob=dropout_prob,
            bias=bias
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size)
        """
        # Pre-layer norm + MLP + residual
        residual = x
        x = self.ln(x)
        x = self.mlp(x)
        x = residual + x
        
        return x


def main():
    """Test the feed forward module."""
    print("[Jadio] Testing Feed Forward module...")
    
    # Test parameters
    batch_size = 2
    seq_len = 10
    hidden_size = 768
    intermediate_size = 3072
    
    # Test MLP
    print("\n--- Testing JadioMLP ---")
    mlp = JadioMLP(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation_function="gelu_new",
        dropout_prob=0.1
    )
    
    # Create test input
    x = torch.randn(batch_size, seq_len, hidden_size)
    print(f"Input shape: {x.shape}")
    print(f"Input mean: {x.mean():.4f}, std: {x.std():.4f}")
    
    # Forward pass
    out = mlp(x)
    print(f"Output shape: {out.shape}")
    print(f"Output mean: {out.mean():.4f}, std: {out.std():.4f}")
    
    # Test different activation functions
    print("\n--- Testing different activation functions ---")
    activations = ["gelu", "gelu_new", "gelu_pytorch_tanh", "relu", "silu"]
    
    for act in activations:
        mlp_act = JadioMLP(hidden_size, intermediate_size, activation_function=act)
        out_act = mlp_act(x)
        print(f"{act}: mean={out_act.mean():.4f}, std={out_act.std():.4f}")
    
    # Test complete feed-forward block
    print("\n--- Testing JadioFeedForward ---")
    ff_block = JadioFeedForward(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation_function="gelu_new",
        dropout_prob=0.1
    )
    
    out_ff = ff_block(x)
    print(f"FeedForward output shape: {out_ff.shape}")
    print(f"FeedForward output mean: {out_ff.mean():.4f}, std: {out_ff.std():.4f}")
    
    # Check residual connection is working
    diff = (out_ff - x).abs().mean()
    print(f"Difference from input (residual effect): {diff:.4f}")
    
    print("\n[Jadio] Feed Forward test completed!")


if __name__ == "__main__":
    main()
"""
Layer normalization module for the Jadio LLM.
"""
import torch
import torch.nn as nn
from typing import Optional


class JadioLayerNorm(nn.Module):
    """
    Layer Normalization for Jadio model.
    
    This implementation follows the standard Layer Normalization paper:
    "Layer Normalization" by Ba et al. (2016)
    """
    
    def __init__(self, normalized_shape: int, eps: float = 1e-5, bias: bool = True):
        """
        Initialize Layer Normalization.
        
        Args:
            normalized_shape: Input shape from an expected input of size
            eps: A value added to the denominator for numerical stability
            bias: If True, adds a learnable bias parameter
        """
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        
        # Learnable parameters
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape)) if bias else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply layer normalization.
        
        Args:
            x: Input tensor of shape (..., normalized_shape)
            
        Returns:
            Normalized tensor of same shape as input
        """
        # Calculate mean and variance along the last dimension
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        
        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # Scale and shift
        x_norm = x_norm * self.weight
        if self.bias is not None:
            x_norm = x_norm + self.bias
            
        return x_norm
    
    def extra_repr(self) -> str:
        """String representation of the layer."""
        return f'normalized_shape={self.normalized_shape}, eps={self.eps}, bias={self.bias is not None}'


class JadioRMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization for Jadio model.
    
    This is an alternative to LayerNorm that only uses RMS normalization
    without mean centering. Often used in newer models like LLaMA.
    """
    
    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        """
        Initialize RMS Normalization.
        
        Args:
            normalized_shape: Input shape from an expected input of size
            eps: A value added to the denominator for numerical stability
        """
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        
        # Learnable scale parameter
        self.weight = nn.Parameter(torch.ones(normalized_shape))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS normalization.
        
        Args:
            x: Input tensor of shape (..., normalized_shape)
            
        Returns:
            Normalized tensor of same shape as input
        """
        # Calculate RMS
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        
        # Normalize and scale
        x_norm = x / rms * self.weight
        
        return x_norm
    
    def extra_repr(self) -> str:
        """String representation of the layer."""
        return f'normalized_shape={self.normalized_shape}, eps={self.eps}'


def create_norm_layer(norm_type: str, normalized_shape: int, eps: float = 1e-5) -> nn.Module:
    """
    Factory function to create normalization layers.
    
    Args:
        norm_type: Type of normalization ('layer_norm' or 'rms_norm')
        normalized_shape: Input shape from an expected input of size
        eps: A value added to the denominator for numerical stability
        
    Returns:
        Normalization layer
    """
    if norm_type.lower() == 'layer_norm':
        return JadioLayerNorm(normalized_shape, eps=eps)
    elif norm_type.lower() == 'rms_norm':
        return JadioRMSNorm(normalized_shape, eps=eps)
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}. Use 'layer_norm' or 'rms_norm'")


def main():
    """Test the layer normalization modules."""
    print("[Jadio] Testing Layer Normalization modules...")
    
    # Test parameters
    batch_size = 2
    seq_len = 10
    hidden_dim = 768
    
    # Create test input
    x = torch.randn(batch_size, seq_len, hidden_dim)
    print(f"Input shape: {x.shape}")
    print(f"Input mean: {x.mean():.4f}, std: {x.std():.4f}")
    
    # Test JadioLayerNorm
    print("\n--- Testing JadioLayerNorm ---")
    layer_norm = JadioLayerNorm(hidden_dim)
    y_ln = layer_norm(x)
    print(f"Output shape: {y_ln.shape}")
    print(f"Output mean: {y_ln.mean():.6f}, std: {y_ln.std():.4f}")
    print(f"Per-token mean (should be ~0): {y_ln.mean(dim=-1).abs().max():.6f}")
    print(f"Per-token std (should be ~1): {y_ln.std(dim=-1).mean():.4f}")
    
    # Test JadioRMSNorm
    print("\n--- Testing JadioRMSNorm ---")
    rms_norm = JadioRMSNorm(hidden_dim)
    y_rms = rms_norm(x)
    print(f"Output shape: {y_rms.shape}")
    print(f"Output mean: {y_rms.mean():.4f}, std: {y_rms.std():.4f}")
    print(f"Per-token RMS: {y_rms.pow(2).mean(dim=-1).sqrt().mean():.4f}")
    
    # Test factory function
    print("\n--- Testing factory function ---")
    ln_factory = create_norm_layer('layer_norm', hidden_dim)
    rms_factory = create_norm_layer('rms_norm', hidden_dim)
    print(f"Created LayerNorm: {type(ln_factory).__name__}")
    print(f"Created RMSNorm: {type(rms_factory).__name__}")
    
    print("\n[Jadio] Layer Normalization tests completed!")


if __name__ == "__main__":
    main()
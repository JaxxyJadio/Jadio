"""
Configuration for the Jadio LLM.
"""
import torch
from dataclasses import dataclass
from typing import Optional

@dataclass
class JadioConfig:
    """Configuration class for Jadio model."""
    
    # Model architecture
    vocab_size: int = 50257  # GPT-2 tokenizer vocab size
    n_positions: int = 1024  # Max sequence length
    n_ctx: int = 1024  # Context length (same as n_positions)
    n_embd: int = 768  # Hidden dimension
    n_layer: int = 12  # Number of transformer layers
    n_head: int = 12  # Number of attention heads
    n_inner: Optional[int] = None  # Inner dimension of feed-forward (4 * n_embd if None)
    
    # Activation and normalization
    activation_function: str = "gelu_new"
    resid_pdrop: float = 0.1  # Residual dropout
    embd_pdrop: float = 0.1  # Embedding dropout
    attn_pdrop: float = 0.1  # Attention dropout
    layer_norm_epsilon: float = 1e-5
    
    # Initialization
    initializer_range: float = 0.02
    scale_attn_weights: bool = True
    use_cache: bool = True
    
    # Training specific
    bos_token_id: int = 50256  # Beginning of sequence token
    eos_token_id: int = 50256  # End of sequence token
    pad_token_id: int = 50256  # Padding token
    
    # Device and dtype
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32
    
    def __post_init__(self):
        """Post-initialization to set derived parameters."""
        if self.n_inner is None:
            self.n_inner = 4 * self.n_embd
        
        # Validate configuration
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
        assert self.vocab_size > 0, "vocab_size must be positive"
        assert self.n_layer > 0, "n_layer must be positive"
        assert self.n_head > 0, "n_head must be positive"
    
    @property
    def head_dim(self) -> int:
        """Head dimension for attention."""
        return self.n_embd // self.n_head
    
    def get_model_params(self) -> int:
        """Calculate approximate number of parameters."""
        # Token embeddings
        token_emb = self.vocab_size * self.n_embd
        
        # Position embeddings
        pos_emb = self.n_positions * self.n_embd
        
        # Transformer layers
        # Each layer has:
        # - Attention: 4 * n_embd^2 (q, k, v, proj)
        # - Feed-forward: 2 * n_embd * n_inner
        # - Layer norms: 2 * n_embd (small, can ignore)
        layer_params = self.n_layer * (4 * self.n_embd**2 + 2 * self.n_embd * self.n_inner)
        
        # Final layer norm and output projection
        final_ln = self.n_embd
        lm_head = self.n_embd * self.vocab_size
        
        total = token_emb + pos_emb + layer_params + final_ln + lm_head
        return total
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "vocab_size": self.vocab_size,
            "n_positions": self.n_positions,
            "n_ctx": self.n_ctx,
            "n_embd": self.n_embd,
            "n_layer": self.n_layer,
            "n_head": self.n_head,
            "n_inner": self.n_inner,
            "activation_function": self.activation_function,
            "resid_pdrop": self.resid_pdrop,
            "embd_pdrop": self.embd_pdrop,
            "attn_pdrop": self.attn_pdrop,
            "layer_norm_epsilon": self.layer_norm_epsilon,
            "initializer_range": self.initializer_range,
            "scale_attn_weights": self.scale_attn_weights,
            "use_cache": self.use_cache,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "pad_token_id": self.pad_token_id,
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "JadioConfig":
        """Create config from dictionary."""
        return cls(**config_dict)


# Predefined configurations
def get_jadio_50m_config() -> JadioConfig:
    """Get configuration for 50M parameter Jadio model."""
    return JadioConfig(
        vocab_size=50257,
        n_positions=1024,
        n_ctx=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        n_inner=3072,  # 4 * 768
        activation_function="gelu_new",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        scale_attn_weights=True,
        use_cache=True,
        bos_token_id=50256,
        eos_token_id=50256,
        pad_token_id=50256,
    )


def get_jadio_small_config() -> JadioConfig:
    """Get configuration for smaller Jadio model (for testing)."""
    return JadioConfig(
        vocab_size=50257,
        n_positions=512,
        n_ctx=512,
        n_embd=384,
        n_layer=6,
        n_head=6,
        n_inner=1536,
        activation_function="gelu_new",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        scale_attn_weights=True,
        use_cache=True,
        bos_token_id=50256,
        eos_token_id=50256,
        pad_token_id=50256,
    )


def main():
    """Test the configuration."""
    config = get_jadio_50m_config()
    print(f"[Jadio] Configuration loaded successfully!")
    print(f"Model parameters: ~{config.get_model_params() / 1e6:.1f}M")
    print(f"Hidden dimension: {config.n_embd}")
    print(f"Layers: {config.n_layer}")
    print(f"Heads: {config.n_head}")
    print(f"Vocabulary size: {config.vocab_size}")
    print(f"Context length: {config.n_ctx}")


if __name__ == "__main__":
    main()
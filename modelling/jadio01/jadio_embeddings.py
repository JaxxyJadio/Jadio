"""
Embeddings module for the Jadio LLM.
"""
import torch
import torch.nn as nn
import math
from typing import Optional


class JadioEmbeddings(nn.Module):
    """
    Token and position embeddings for Jadio.
    
    Combines token embeddings with learned positional embeddings,
    following the GPT-2 style approach.
    """
    
    def __init__(self, 
                 vocab_size: int,
                 hidden_size: int,
                 max_position_embeddings: int,
                 dropout_prob: float = 0.1,
                 pad_token_id: Optional[int] = None):
        """
        Initialize the embeddings.
        
        Args:
            vocab_size: Size of the vocabulary
            hidden_size: Hidden dimension size
            max_position_embeddings: Maximum sequence length
            dropout_prob: Dropout probability
            pad_token_id: ID of the padding token
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.pad_token_id = pad_token_id
        
        # Token embeddings
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        
        # Position embeddings (learned)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_prob)
        
        # Register position_ids as buffer
        self.register_buffer(
            "position_ids",
            torch.arange(max_position_embeddings, dtype=torch.long).unsqueeze(0),
            persistent=False
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize embedding weights."""
        # Token embeddings - normal initialization
        nn.init.normal_(self.token_embeddings.weight, std=0.02)
        if self.pad_token_id is not None:
            # Zero out padding token embedding
            with torch.no_grad():
                self.token_embeddings.weight[self.pad_token_id].fill_(0)
        
        # Position embeddings - normal initialization
        nn.init.normal_(self.position_embeddings.weight, std=0.02)
    
    def forward(self, 
                input_ids: torch.Tensor,
                position_ids: Optional[torch.Tensor] = None,
                past_key_values_length: int = 0) -> torch.Tensor:
        """
        Forward pass of the embeddings.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            position_ids: Position IDs of shape (batch_size, seq_len). If None, auto-generated.
            past_key_values_length: Length of past key values for position calculation
            
        Returns:
            Combined embeddings of shape (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len = input_ids.shape
        
        # Get token embeddings
        token_embeds = self.token_embeddings(input_ids)
        
        # Get position IDs if not provided
        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length:past_key_values_length + seq_len]
            position_ids = position_ids.expand(batch_size, -1)
        
        # Get position embeddings
        position_embeds = self.position_embeddings(position_ids)
        
        # Combine embeddings
        embeddings = token_embeds + position_embeds
        
        # Apply dropout
        embeddings = self.dropout(embeddings)
        
        return embeddings
    
    def get_input_embeddings(self) -> nn.Embedding:
        """Get the token embedding layer."""
        return self.token_embeddings
    
    def set_input_embeddings(self, new_embeddings: nn.Embedding):
        """Set new token embedding layer."""
        self.token_embeddings = new_embeddings


class JadioLMHead(nn.Module):
    """
    Language modeling head for Jadio.
    
    Projects hidden states back to vocabulary size for next token prediction.
    """
    
    def __init__(self, hidden_size: int, vocab_size: int, bias: bool = False):
        """
        Initialize the LM head.
        
        Args:
            hidden_size: Hidden dimension size
            vocab_size: Size of the vocabulary
            bias: Whether to use bias in the linear layer
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        # Linear projection to vocabulary
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=bias)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize the weights."""
        nn.init.normal_(self.lm_head.weight, std=0.02)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the LM head.
        
        Args:
            hidden_states: Hidden states of shape (batch_size, seq_len, hidden_size)
            
        Returns:
            Logits of shape (batch_size, seq_len, vocab_size)
        """
        logits = self.lm_head(hidden_states)
        return logits


def main():
    """Test the embeddings module."""
    print("[Jadio] Testing Embeddings module...")
    
    # Test parameters
    batch_size = 2
    seq_len = 10
    vocab_size = 50257
    hidden_size = 768
    max_pos_emb = 1024
    
    # Create embeddings
    embeddings = JadioEmbeddings(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        max_position_embeddings=max_pos_emb,
        dropout_prob=0.1,
        pad_token_id=50256
    )
    
    # Create LM head
    lm_head = JadioLMHead(hidden_size, vocab_size)
    
    # Create test input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    print(f"Input IDs shape: {input_ids.shape}")
    
    # Forward pass through embeddings
    embeds = embeddings(input_ids)
    print(f"Embeddings shape: {embeds.shape}")
    print(f"Embeddings mean: {embeds.mean():.4f}, std: {embeds.std():.4f}")
    
    # Forward pass through LM head
    logits = lm_head(embeds)
    print(f"Logits shape: {logits.shape}")
    print(f"Logits mean: {logits.mean():.4f}, std: {logits.std():.4f}")
    
    # Test with custom position IDs
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    embeds_custom_pos = embeddings(input_ids, position_ids=position_ids)
    print(f"Custom position embeddings shape: {embeds_custom_pos.shape}")
    
    # Check that embeddings are different for different positions
    print(f"Position effect - difference between pos 0 and pos 1: {(embeds[0, 1] - embeds[0, 0]).abs().mean():.4f}")
    
    print("[Jadio] Embeddings test completed!")


if __name__ == "__main__":
    main()
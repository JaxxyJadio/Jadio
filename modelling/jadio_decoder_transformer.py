"""
Jadio Decoder Transformer - Main Model Implementation
GPT-2 style decoder-only transformer for the Jadio LLM.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Union, List
from dataclasses import dataclass

# Import our modules
from .jadio_attention import JadioAttention
from .jadio_feed_forward import JadioMLP
from .jadio_embeddings import JadioEmbeddings, JadioLMHead
from .jadio_layer_norm import JadioLayerNorm


@dataclass
class CausalLMOutput:
    """Output type for causal language modeling."""
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class JadioBlock(nn.Module):
    """
    Single transformer block combining attention and feed-forward layers.
    
    Architecture (pre-norm):
    x -> LayerNorm -> SelfAttention -> (+) -> LayerNorm -> FeedForward -> (+)
    |                                  |     |                            |
    +----------------------------------+     +----------------------------+
    """
    
    def __init__(self, config: JadioConfig, layer_idx: int = 0):
        """
        Initialize transformer block.
        
        Args:
            config: Model configuration
            layer_idx: Layer index (for debugging/logging)
        """
        super().__init__()
        self.layer_idx = layer_idx
        hidden_size = config.n_embd
        
        # Self-attention with pre-norm
        self.ln_1 = JadioLayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = JadioAttention(
            hidden_size=hidden_size,
            num_attention_heads=config.n_head,
            attention_dropout=config.attn_pdrop,
            resid_dropout=config.resid_pdrop,
            max_position_embeddings=config.n_positions,
            scale_attn_weights=config.scale_attn_weights,
            bias=True
        )
        
        # Feed-forward with pre-norm
        self.ln_2 = JadioLayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = JadioMLP(
            hidden_size=hidden_size,
            intermediate_size=config.n_inner,
            activation_function=config.activation_function,
            dropout_prob=config.resid_pdrop,
            bias=True
        )
    
    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                use_cache: bool = False,
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                output_attentions: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Forward pass of transformer block.
        
        Args:
            hidden_states: Input tensor (batch_size, seq_len, hidden_size)
            attention_mask: Optional attention mask
            use_cache: Whether to return key/value for caching
            past_key_value: Cached key/value from previous steps
            output_attentions: Whether to return attention weights
            
        Returns:
            Output tensor or tuple with additional outputs
        """
        residual = hidden_states
        
        # Pre-norm + Self-attention
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            attention_mask=attention_mask,
            use_cache=use_cache,
            past_key_value=past_key_value,
            output_attentions=output_attentions
        )
        
        if isinstance(attn_outputs, tuple):
            attn_output = attn_outputs[0]
            outputs = attn_outputs[1:]
        else:
            attn_output = attn_outputs
            outputs = ()
        
        # Residual connection
        hidden_states = residual + attn_output
        
        # Pre-norm + Feed-forward
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        
        # Residual connection
        hidden_states = residual + hidden_states
        
        if outputs:
            return (hidden_states,) + outputs
        else:
            return hidden_states


class JadioModel(nn.Module):
    """
    Core Jadio transformer model (without language modeling head).
    
    This is the base transformer that can be used for various tasks
    by adding different heads on top.
    """
    
    def __init__(self, config: JadioConfig):
        """
        Initialize the Jadio model.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        
        # Embeddings
        self.wte = JadioEmbeddings(
            vocab_size=config.vocab_size,
            hidden_size=config.n_embd,
            max_position_embeddings=config.n_positions,
            dropout_prob=config.embd_pdrop,
            pad_token_id=config.pad_token_id
        )
        
        # Transformer blocks
        self.h = nn.ModuleList([
            JadioBlock(config, layer_idx=i) 
            for i in range(config.n_layer)
        ])
        
        # Final layer norm
        self.ln_f = JadioLayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        
        # Dropout
        self.drop = nn.Dropout(config.embd_pdrop)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Apply special scaled init to the residual projections
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
    
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, JadioLayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def get_input_embeddings(self):
        """Get input embedding layer."""
        return self.wte.token_embeddings
    
    def set_input_embeddings(self, new_embeddings):
        """Set input embedding layer."""
        self.wte.token_embeddings = new_embeddings
    
    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Forward pass of the model.
        
        Args:
            input_ids: Token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            position_ids: Position IDs (batch_size, seq_len)
            past_key_values: Cached key/values from previous steps
            use_cache: Whether to return key/value for caching
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return all hidden states
            
        Returns:
            Model outputs
        """
        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        
        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        
        # Embeddings
        hidden_states = self.wte(
            input_ids,
            position_ids=position_ids,
            past_key_values_length=past_length
        )
        
        # Dropout
        hidden_states = self.drop(hidden_states)
        
        # Prepare outputs
        presents = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        
        # Forward through transformer blocks
        for i, (block, past_key_value) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            outputs = block(
                hidden_states,
                attention_mask=attention_mask,
                use_cache=use_cache,
                past_key_value=past_key_value,
                output_attentions=output_attentions
            )
            
            if isinstance(outputs, tuple):
                hidden_states = outputs[0]
                if use_cache:
                    presents = presents + (outputs[1],)
                if output_attentions:
                    all_attentions = all_attentions + (outputs[2 if use_cache else 1],)
            else:
                hidden_states = outputs
        
        # Final layer norm
        hidden_states = self.ln_f(hidden_states)
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        # Return outputs
        if not any([use_cache, output_attentions, output_hidden_states]):
            return hidden_states
        
        return {
            'last_hidden_state': hidden_states,
            'past_key_values': presents,
            'hidden_states': all_hidden_states,
            'attentions': all_attentions
        }


class JadioLMHeadModel(nn.Module):
    """
    Jadio Model with Language Modeling Head.
    
    This is the complete model for causal language modeling tasks.
    """
    
    def __init__(self, config: JadioConfig):
        """
        Initialize the LM head model.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        
        # Base transformer
        self.transformer = JadioModel(config)
        
        # Language modeling head
        self.lm_head = JadioLMHead(config.n_embd, config.vocab_size, bias=False)
        
        # Tie weights between input embeddings and output projection
        # This is a common practice in language models
        self.tie_weights()
    
    def tie_weights(self):
        """Tie input and output embedding weights."""
        self.lm_head.lm_head.weight = self.transformer.wte.token_embeddings.weight
    
    def get_input_embeddings(self):
        """Get input embedding layer."""
        return self.transformer.get_input_embeddings()
    
    def set_input_embeddings(self, new_embeddings):
        """Set input embedding layer."""
        self.transformer.set_input_embeddings(new_embeddings)
        
    def get_output_embeddings(self):
        """Get output embedding layer."""
        return self.lm_head
    
    def set_output_embeddings(self, new_embeddings):
        """Set output embedding layer."""
        self.lm_head = new_embeddings
    
    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wte.position_embeddings.weight.numel()
        return n_params
    
    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                labels: Optional[torch.Tensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None) -> Union[torch.Tensor, CausalLMOutput]:
        """
        Forward pass for language modeling.
        
        Args:
            input_ids: Token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            position_ids: Position IDs (batch_size, seq_len)
            past_key_values: Cached key/values from previous steps
            labels: Target token IDs for computing loss (batch_size, seq_len)
            use_cache: Whether to return key/value for caching
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return all hidden states
            
        Returns:
            Loss and logits if labels provided, otherwise just logits
        """
        # Forward through transformer
        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )
        
        if isinstance(transformer_outputs, dict):
            hidden_states = transformer_outputs['last_hidden_state']
        else:
            hidden_states = transformer_outputs
        
        # Language modeling head
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        if not use_cache and not output_attentions and not output_hidden_states:
            if labels is not None:
                return loss, logits
            return logits
        
        # Return structured output
        return CausalLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=transformer_outputs.get('past_key_values') if isinstance(transformer_outputs, dict) else None,
            hidden_states=transformer_outputs.get('hidden_states') if isinstance(transformer_outputs, dict) else None,
            attentions=transformer_outputs.get('attentions') if isinstance(transformer_outputs, dict) else None,
        )
    
    @torch.no_grad()
    def generate(self,
                 input_ids: torch.Tensor,
                 max_new_tokens: int = 50,
                 temperature: float = 1.0,
                 top_k: Optional[int] = None,
                 top_p: Optional[float] = None,
                 do_sample: bool = True,
                 pad_token_id: Optional[int] = None,
                 eos_token_id: Optional[int] = None) -> torch.Tensor:
        """
        Generate text using the model.
        
        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Keep only top k tokens for sampling
            top_p: Keep only top p probability mass for sampling
            do_sample: Whether to sample or use greedy decoding
            pad_token_id: Padding token ID
            eos_token_id: End of sequence token ID
            
        Returns:
            Generated token IDs (batch_size, seq_len + max_new_tokens)
        """
        self.eval()
        device = input_ids.device
        batch_size, seq_len = input_ids.shape
        
        # Set default pad/eos tokens
        if pad_token_id is None:
            pad_token_id = self.config.pad_token_id
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id
        
        # Track which sequences are finished
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=device)
        
        # Generate tokens
        for _ in range(max_new_tokens):
            # Forward pass
            outputs = self(input_ids, use_cache=False)
            if isinstance(outputs, tuple):
                logits = outputs[1] if len(outputs) > 1 else outputs[0]
            else:
                logits = outputs
            
            # Get logits for last token
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Scatter sorted tensors to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample or use greedy decoding
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Update only unfinished sequences
            next_tokens = next_tokens * unfinished_sequences.unsqueeze(-1) + pad_token_id * (1 - unfinished_sequences.unsqueeze(-1))
            
            # Concatenate new token
            input_ids = torch.cat([input_ids, next_tokens], dim=-1)
            
            # Check if we hit EOS token
            if eos_token_id is not None:
                unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).prod(dim=-1))
            
            # Stop if all sequences are finished
            if unfinished_sequences.max() == 0:
                break
        
        return input_ids


def main():
    """Test the decoder transformer."""
    print("[Jadio] Testing Decoder Transformer...")
    
    # Import config here to avoid circular imports in main execution
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config.jadio_config import get_jadio_small_config
    
    # Use small config for testing
    config = get_jadio_small_config()
    print(f"Config: {config.n_layer} layers, {config.n_embd} hidden, {config.n_head} heads")
    
    # Create model
    model = JadioLMHeadModel(config)
    print(f"Model parameters: {model.get_num_params() / 1e6:.1f}M")
    
    # Test input
    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    # Test forward pass
    print("\n--- Testing forward pass ---")
    model.eval()
    with torch.no_grad():
        logits = model(input_ids)
        print(f"Logits shape: {logits.shape}")
        print(f"Logits range: [{logits.min():.2f}, {logits.max():.2f}]")
    
    # Test with labels (training mode)
    print("\n--- Testing training mode ---")
    model.train()
    labels = input_ids.clone()
    loss, logits = model(input_ids, labels=labels)
    print(f"Training loss: {loss.item():.4f}")
    print(f"Loss is finite: {torch.isfinite(loss).item()}")
    
    # Test generation
    print("\n--- Testing generation ---")
    model.eval()
    prompt = input_ids[:1, :8]  # First sample, first 8 tokens
    generated = model.generate(
        prompt,
        max_new_tokens=10,
        temperature=0.8,
        do_sample=True
    )
    print(f"Generated shape: {generated.shape}")
    print(f"Generated {generated.shape[1] - prompt.shape[1]} new tokens")
    
    print("\n[Jadio] Decoder Transformer test completed!")


if __name__ == "__main__":
    main()
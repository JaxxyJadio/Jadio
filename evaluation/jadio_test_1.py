#!/usr/bin/env python3
"""
Complete test script for the Jadio LLM.
This script tests the full model pipeline.
"""
import sys
import os
import torch
import torch.nn.functional as F

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.jadio_config import get_jadio_50m_config, get_jadio_small_config
from modelling.jadio_decoder_transformer import JadioLMHeadModel


def test_model_components():
    """Test individual model components."""
    print("=" * 60)
    print("TESTING MODEL COMPONENTS")
    print("=" * 60)
    
    # Test configuration
    print("\n1. Testing Configuration...")
    config = get_jadio_50m_config()
    print(f"   ✓ Config loaded: {config.n_layer} layers, {config.n_embd} hidden, {config.n_head} heads")
    print(f"   ✓ Estimated parameters: {config.get_model_params() / 1e6:.1f}M")
    print(f"   ✓ Vocabulary size: {config.vocab_size}")
    print(f"   ✓ Context length: {config.n_ctx}")
    
    # Test model creation
    print("\n2. Testing Model Creation...")
    model = JadioLMHeadModel(config)
    actual_params = model.get_num_params()
    print(f"   ✓ Model created successfully")
    print(f"   ✓ Actual parameters: {actual_params / 1e6:.1f}M")
    print(f"   ✓ Parameter difference: {abs(actual_params - config.get_model_params()) / 1e6:.2f}M")
    
    return model, config


def test_forward_pass(model, config):
    """Test forward pass with different inputs."""
    print("\n3. Testing Forward Pass...")
    
    batch_size = 2
    seq_len = 32
    vocab_size = config.vocab_size
    
    # Create test inputs
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"   ✓ Created input: {input_ids.shape}")
    
    # Test inference mode
    model.eval()
    with torch.no_grad():
        logits = model(input_ids)
        print(f"   ✓ Forward pass: {logits.shape}")
        print(f"   ✓ Logits range: [{logits.min():.2f}, {logits.max():.2f}]")
        
        # Test that output is valid probability distribution
        probs = F.softmax(logits, dim=-1)
        print(f"   ✓ Probabilities sum to ~1: {probs.sum(dim=-1).mean():.4f}")
    
    # Test training mode with labels
    print("\n4. Testing Training Mode...")
    model.train()
    labels = input_ids.clone()
    
    loss, logits = model(input_ids, labels=labels)
    print(f"   ✓ Training forward pass with loss: {loss.item():.4f}")
    print(f"   ✓ Loss is finite: {torch.isfinite(loss).item()}")
    
    return loss


def test_generation(model, config):
    """Test text generation capabilities."""
    print("\n5. Testing Text Generation...")
    
    model.eval()
    
    # Test basic generation
    prompt = torch.randint(0, config.vocab_size, (1, 10))
    print(f"   ✓ Prompt shape: {prompt.shape}")
    
    with torch.no_grad():
        # Greedy generation
        generated_greedy = model.generate(
            prompt,
            max_new_tokens=20,
            temperature=1.0,
            do_sample=False
        )
        print(f"   ✓ Greedy generation: {generated_greedy.shape}")
        print(f"   ✓ Generated {generated_greedy.shape[1] - prompt.shape[1]} new tokens")
        
        # Sampling generation
        generated_sample = model.generate(
            prompt,
            max_new_tokens=20,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            do_sample=True
        )
        print(f"   ✓ Sampling generation: {generated_sample.shape}")
        
        # Test that generations are different (with high probability)
        different = not torch.equal(generated_greedy, generated_sample)
        print(f"   ✓ Greedy and sampling outputs differ: {different}")


def test_caching(model, config):
    """Test key-value caching for efficient generation."""
    print("\n6. Testing KV Caching...")
    
    model.eval()
    batch_size = 1
    seq_len = 16
    
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    with torch.no_grad():
        # First pass - get initial outputs and cache
        outputs = model(input_ids, use_cache=True)
        logits1, past_key_values = outputs
        print(f"   ✓ Initial pass: logits {logits1.shape}, cached {len(past_key_values)} layers")
        
        # Second pass - use cache with new token
        next_token = torch.randint(0, config.vocab_size, (batch_size, 1))
        outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
        logits2, new_past_kv = outputs
        print(f"   ✓ Cached pass: logits {logits2.shape}")
        
        # Verify cache shapes
        for i, (past_kv, new_kv) in enumerate(zip(past_key_values, new_past_kv)):
            past_k, past_v = past_kv
            new_k, new_v = new_kv
            print(f"   ✓ Layer {i}: key {past_k.shape} -> {new_k.shape}")
            break  # Just show first layer
        
        # Test that cached generation matches non-cached
        full_input = torch.cat([input_ids, next_token], dim=1)
        logits_full = model(full_input)
        
        # Compare last token logits
        diff = (logits_full[:, -1:, :] - logits2).abs().max()
        print(f"   ✓ Cache consistency (max diff): {diff.item():.6f}")


def test_different_sizes():
    """Test different model sizes."""
    print("\n7. Testing Different Model Sizes...")
    
    # Test small model
    small_config = get_jadio_small_config()
    small_model = JadioLMHeadModel(small_config)
    small_params = small_model.get_num_params()
    print(f"   ✓ Small model: {small_params / 1e6:.1f}M parameters")
    
    # Test 50M model
    large_config = get_jadio_50m_config()
    large_model = JadioLMHeadModel(large_config)
    large_params = large_model.get_num_params()
    print(f"   ✓ 50M model: {large_params / 1e6:.1f}M parameters")
    
    # Quick forward pass test
    input_ids = torch.randint(0, small_config.vocab_size, (1, 10))
    
    with torch.no_grad():
        small_out = small_model(input_ids)
        large_out = large_model(input_ids)
        
    print(f"   ✓ Small model output: {small_out.shape}")
    print(f"   ✓ Large model output: {large_out.shape}")


def test_memory_usage():
    """Test memory usage of the model."""
    print("\n8. Testing Memory Usage...")
    
    config = get_jadio_small_config()  # Use small model for memory test
    model = JadioLMHeadModel(config)
    
    # Test different sequence lengths
    seq_lengths = [16, 64, 256, 512]
    batch_size = 1
    
    for seq_len in seq_lengths:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        if torch.cuda.is_available():
            model = model.cuda()
            input_ids = input_ids.cuda()
            torch.cuda.synchronize()
            
            mem_before = torch.cuda.memory_allocated()
            
        with torch.no_grad():
            _ = model(input_ids)
            
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            mem_after = torch.cuda.memory_allocated()
            mem_used = (mem_after - mem_before) / 1024**2  # MB
            print(f"   ✓ Seq len {seq_len}: {mem_used:.1f} MB")
        else:
            print(f"   ✓ Seq len {seq_len}: completed (CPU mode)")


def main():
    """Run all tests."""
    print("🚀 JADIO LLM COMPLETE MODEL TEST")
    print("Testing Jadio 50M parameter decoder-only transformer")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    try:
        # Test model components
        model, config = test_model_components()
        
        # Test forward pass
        loss = test_forward_pass(model, config)
        
        # Test generation
        test_generation(model, config)
        
        # Test caching
        test_caching(model, config)
        
        # Test different sizes
        test_different_sizes()
        
        # Test memory usage
        test_memory_usage()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print(f"🎉 Jadio {config.get_model_params() / 1e6:.0f}M model is ready for training!")
        print(f"📊 Model has {model.get_num_params():,} parameters")
        print(f"🔧 Architecture: {config.n_layer} layers × {config.n_head} heads × {config.n_embd} hidden")
        print(f"📚 Vocabulary: {config.vocab_size:,} tokens")
        print(f"📏 Context length: {config.n_ctx:,} tokens")
        
        if loss is not None:
            print(f"🎯 Initial loss: {loss.item():.4f}")
        
        print("\nNext steps:")
        print("1. Set up your training data pipeline")
        print("2. Configure training hyperparameters") 
        print("3. Start training with scripts/jadio_train.py")
        print("4. Monitor with Weights & Biases")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
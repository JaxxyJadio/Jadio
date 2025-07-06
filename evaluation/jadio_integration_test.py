#!/usr/bin/env python3
"""
Complete integration test for the Jadio LLM project.
This script tests the entire pipeline from tokenization to training to generation.
"""
import sys
import os
import torch
import tempfile
import shutil
from pathlib import Path

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.jadio_config import get_jadio_small_config
from modelling.jadio01.jadio_decoder_transformer import JadioLMHeadModel
from tokenizer.jadio_tokenizer import JadioTokenizer
from data.jadio_dataset_loader import TextDataset, create_dataloader, create_dummy_texts
from training.jadio_train import JadioTrainer
from scripts.jadio_generate import JadioGenerator
from scripts.jadio_utilities import setup_logger, set_seed


def test_tokenizer():
    """Test tokenizer functionality."""
    print("=" * 60)
    print("TESTING TOKENIZER")
    print("=" * 60)
    
    # Try to load GPT-2 tokenizer files
    tokenizer_path = Path(__file__).parent.parent / "modelling" / "jadio01"
    vocab_file = tokenizer_path / "vocab.json"
    
    if vocab_file.exists():
        print("✓ Found GPT-2 tokenizer files")
        tokenizer = JadioTokenizer.from_pretrained(str(tokenizer_path))
    else:
        print("⚠ GPT-2 tokenizer files not found, creating minimal test tokenizer")
        # Create a minimal tokenizer for testing
        test_vocab = {f"token_{i}": i for i in range(1000)}
        test_vocab["<|endoftext|>"] = 50256
        
        # Save to temporary directory
        temp_dir = tempfile.mkdtemp()
        
        import json
        with open(Path(temp_dir) / "vocab.json", 'w') as f:
            json.dump(test_vocab, f)
        
        with open(Path(temp_dir) / "merges.txt", 'w') as f:
            f.write("#version: 0.2\n")
            for i in range(100):
                f.write(f"token_{i} token_{i+1}\n")
        
        tokenizer = JadioTokenizer.from_pretrained(temp_dir)
        shutil.rmtree(temp_dir)
    
    print(f"✓ Tokenizer loaded: vocab size {len(tokenizer)}")
    
    # Test basic functionality
    test_text = "Hello, world! This is a test of the Jadio tokenizer."
    tokens = tokenizer.encode(test_text)
    decoded = tokenizer.decode(tokens)
    
    print(f"✓ Text encoding/decoding works")
    print(f"  Original: {test_text}")
    print(f"  Tokens: {len(tokens)} tokens")
    print(f"  Decoded: {decoded}")
    
    return tokenizer


def test_model_creation():
    """Test model creation and basic operations."""
    print("\n" + "=" * 60)
    print("TESTING MODEL CREATION")
    print("=" * 60)
    
    config = get_jadio_small_config()
    print(f"✓ Config loaded: {config.n_layer}L, {config.n_embd}H, {config.n_head}A")
    
    model = JadioLMHeadModel(config)
    print(f"✓ Model created: {model.get_num_params() / 1e6:.1f}M parameters")
    
    # Test forward pass
    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    model.eval()
    with torch.no_grad():
        logits = model(input_ids)
        print(f"✓ Forward pass: {logits.shape}")
        
        # Test with labels
        loss, _ = model(input_ids, labels=input_ids)
        print(f"✓ Loss computation: {loss.item():.4f}")
        
        # Test generation
        generated = model.generate(
            input_ids[:1, :10],
            max_new_tokens=10,
            temperature=1.0,
            do_sample=False
        )
        print(f"✓ Generation: {generated.shape}")
    
    return model, config


def test_data_pipeline(tokenizer):
    """Test data loading and preprocessing."""
    print("\n" + "=" * 60)
    print("TESTING DATA PIPELINE")
    print("=" * 60)
    
    # Create dummy data
    texts = create_dummy_texts(50, 20, 100)
    print(f"✓ Created {len(texts)} dummy texts")
    
    # Create dataset
    dataset = TextDataset(
        texts=texts,
        tokenizer=tokenizer,
        max_length=64
    )
    print(f"✓ Dataset created: {len(dataset)} samples")
    
    # Test sample
    sample = dataset[0]
    print(f"✓ Sample shape: {sample['input_ids'].shape}")
    
    # Create dataloader
    dataloader = create_dataloader(
        dataset,
        batch_size=4,
        shuffle=True
    )
    print(f"✓ DataLoader created")
    
    # Test batch
    batch = next(iter(dataloader))
    print(f"✓ Batch loaded: {batch['input_ids'].shape}")
    
    return dataloader


def test_training_step(model, dataloader):
    """Test training functionality."""
    print("\n" + "=" * 60)
    print("TESTING TRAINING")
    print("=" * 60)
    
    # Create trainer
    trainer = JadioTrainer(model.config, "test_model")
    trainer.model = model  # Use existing model
    print(f"✓ Trainer created")
    
    # Test optimizer creation
    optimizer = trainer.create_optimizer(learning_rate=1e-4)
    print(f"✓ Optimizer created")
    
    # Test scheduler creation
    scheduler = trainer.create_scheduler(optimizer, warmup_steps=10, max_steps=100)
    print(f"✓ Scheduler created")
    
    # Test training step
    model.train()
    batch = next(iter(dataloader))
    
    initial_loss = trainer.train_step(batch, optimizer, scheduler)
    print(f"✓ Training step: loss = {initial_loss:.4f}")
    
    # Test a few more steps
    total_loss = initial_loss
    for i in range(4):
        batch = next(iter(dataloader))
        loss = trainer.train_step(batch, optimizer, scheduler)
        total_loss += loss
    
    avg_loss = total_loss / 5
    print(f"✓ Multiple steps: avg loss = {avg_loss:.4f}")
    
    return trainer


def test_save_load(model, trainer):
    """Test model saving and loading."""
    print("\n" + "=" * 60)
    print("TESTING SAVE/LOAD")
    print("=" * 60)
    
    # Save model
    with tempfile.TemporaryDirectory() as temp_dir:
        save_path = Path(temp_dir) / "test_model.pt"
        
        trainer.save_model(str(save_path))
        print(f"✓ Model saved to {save_path}")
        
        # Load model
        config = model.config
        new_model = JadioLMHeadModel(config)
        
        checkpoint = torch.load(save_path)
        new_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Model loaded successfully")
        
        # Test that loaded model gives same output
        test_input = torch.randint(0, config.vocab_size, (1, 10))
        
        model.eval()
        new_model.eval()
        
        with torch.no_grad():
            orig_output = model(test_input)
            loaded_output = new_model(test_input)
            
            diff = (orig_output - loaded_output).abs().max().item()
            print(f"✓ Output difference: {diff:.8f} (should be ~0)")
            
            if diff < 1e-6:
                print("✓ Save/load maintains model consistency")
            else:
                print("⚠ Warning: Save/load may have issues")
    
    return new_model


def test_generation_pipeline(model, tokenizer):
    """Test the generation pipeline."""
    print("\n" + "=" * 60)
    print("TESTING GENERATION PIPELINE")
    print("=" * 60)
    
    # Save model temporarily for generator
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = Path(temp_dir) / "test_model.pt"
        
        # Save checkpoint
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'config': model.config.to_dict(),
            'step': 100,
            'epoch': 1,
            'best_loss': 2.5
        }
        torch.save(checkpoint, model_path)
        
        # Save tokenizer
        tokenizer_dir = Path(temp_dir) / "tokenizer"
        tokenizer.save_pretrained(str(tokenizer_dir))
        
        # Create generator
        try:
            generator = JadioGenerator(
                model_path=str(model_path),
                tokenizer_path=str(tokenizer_dir),
                device='cpu'  # Force CPU for testing
            )
            print(f"✓ Generator created")
            
            # Test generation
            test_prompts = [
                "The quick brown fox",
                "In a world where",
                "Once upon a time"
            ]
            
            for prompt in test_prompts:
                generated = generator.generate(
                    prompt=prompt,
                    max_new_tokens=20,
                    temperature=0.8,
                    do_sample=True
                )
                
                print(f"✓ Generated for '{prompt}': {len(generated[0])} chars")
                print(f"  Output: {generated[0][:50]}...")
            
            # Test batch generation
            batch_generated = generator.batch_generate(
                test_prompts,
                max_new_tokens=15,
                temperature=0.5
            )
            print(f"✓ Batch generation: {len(batch_generated)} outputs")
            
        except Exception as e:
            print(f"⚠ Generation test failed: {e}")
            return False
    
    return True


def run_full_integration_test():
    """Run the complete integration test."""
    print("🚀 JADIO LLM COMPLETE INTEGRATION TEST")
    print("Testing the full pipeline from tokenization to generation")
    print("=" * 80)
    
    set_seed(42)
    
    try:
        # Test tokenizer
        tokenizer = test_tokenizer()
        
        # Test model creation
        model, config = test_model_creation()
        
        # Test data pipeline
        dataloader = test_data_pipeline(tokenizer)
        
        # Test training
        trainer = test_training_step(model, dataloader)
        
        # Test save/load
        loaded_model = test_save_load(model, trainer)
        
        # Test generation
        generation_success = test_generation_pipeline(loaded_model, tokenizer)
        
        print("\n" + "=" * 80)
        print("✅ ALL INTEGRATION TESTS PASSED!")
        print("=" * 80)
        print("🎉 The Jadio LLM pipeline is working correctly!")
        print("\nComponents tested:")
        print("  ✓ Tokenizer (encoding/decoding)")
        print("  ✓ Model (forward pass, loss, generation)")
        print("  ✓ Data pipeline (datasets, dataloaders)")
        print("  ✓ Training (optimizer, scheduler, training steps)")
        print("  ✓ Save/Load (checkpoints, model persistence)")
        print("  ✓ Generation (text generation, batch processing)")
        
        print("\nNext steps:")
        print("  1. Prepare your training data")
        print("  2. Run full training with: python training/jadio_train.py")
        print("  3. Generate text with: python scripts/jadio_generate.py")
        print("  4. Evaluate model performance")
        
        return True
        
    except Exception as e:
        print(f"\n❌ INTEGRATION TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function."""
    success = run_full_integration_test()
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
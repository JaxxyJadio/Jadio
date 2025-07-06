#!/usr/bin/env python3
"""
Text generation script for the Jadio LLM.
"""
import os
import sys
import torch
import argparse
import time
from pathlib import Path
from typing import List, Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.jadio_config import JadioConfig, get_jadio_50m_config, get_jadio_small_config
from modelling.jadio01.jadio_decoder_transformer import JadioLMHeadModel
from tokenizer.jadio_tokenizer import JadioTokenizer
from scripts.jadio_utilities import setup_logger, set_seed


class JadioGenerator:
    """Text generator for Jadio LLM."""
    
    def __init__(self, 
                 model_path: str,
                 tokenizer_path: Optional[str] = None,
                 device: str = 'auto'):
        """
        Initialize the generator.
        
        Args:
            model_path: Path to saved model checkpoint
            tokenizer_path: Path to tokenizer files (default: same as model)
            device: Device to use ('auto', 'cuda', 'cpu')
        """
        self.logger = setup_logger("jadio_generator")
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.logger.info(f"Using device: {self.device}")
        
        # Load model
        self.logger.info(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Load config
        if 'config' in checkpoint:
            config_dict = checkpoint['config']
            self.config = JadioConfig.from_dict(config_dict)
        else:
            self.logger.warning("No config in checkpoint, using default")
            self.config = get_jadio_50m_config()
        
        # Initialize model
        self.model = JadioLMHeadModel(self.config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        self.logger.info(f"Model loaded: {self.model.get_num_params() / 1e6:.1f}M parameters")
        
        # Load tokenizer
        tokenizer_path = tokenizer_path or os.path.dirname(model_path)
        self.logger.info(f"Loading tokenizer from {tokenizer_path}")
        
        try:
            self.tokenizer = JadioTokenizer.from_pretrained(tokenizer_path)
            self.logger.info(f"Tokenizer loaded: vocab size {len(self.tokenizer)}")
        except Exception as e:
            self.logger.warning(f"Could not load tokenizer: {e}")
            self.logger.info("Using default tokenizer path")
            # Try default path
            default_tokenizer_path = Path(__file__).parent.parent / "modelling" / "jadio01"
            self.tokenizer = JadioTokenizer.from_pretrained(str(default_tokenizer_path))
    
    @torch.no_grad()
    def generate(self,
                 prompt: str,
                 max_new_tokens: int = 100,
                 temperature: float = 1.0,
                 top_k: Optional[int] = 50,
                 top_p: Optional[float] = 0.9,
                 do_sample: bool = True,
                 num_return_sequences: int = 1,
                 repetition_penalty: float = 1.0,
                 length_penalty: float = 1.0,
                 early_stopping: bool = True,
                 pad_token_id: Optional[int] = None,
                 eos_token_id: Optional[int] = None) -> List[str]:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Keep only top k tokens for sampling
            top_p: Keep only top p probability mass for sampling
            do_sample: Whether to sample or use greedy decoding
            num_return_sequences: Number of sequences to generate
            repetition_penalty: Penalty for repeating tokens
            length_penalty: Penalty for sequence length
            early_stopping: Whether to stop at EOS token
            pad_token_id: Padding token ID
            eos_token_id: End of sequence token ID
            
        Returns:
            List of generated text strings
        """
        # Set default token IDs
        if pad_token_id is None:
            pad_token_id = self.tokenizer.pad_token_id
        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_token_id
        
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        
        # Repeat for multiple sequences
        if num_return_sequences > 1:
            input_ids = input_ids.repeat(num_return_sequences, 1)
        
        # Generate
        generated_sequences = []
        for i in range(num_return_sequences):
            current_input = input_ids[i:i+1] if num_return_sequences > 1 else input_ids
            
            generated = self.model.generate(
                current_input,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id if early_stopping else None
            )
            
            # Decode generated sequence
            generated_tokens = generated[0].cpu().tolist()
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Remove the original prompt from the generated text
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            generated_sequences.append(generated_text)
        
        return generated_sequences
    
    def interactive_generation(self):
        """Interactive text generation loop."""
        print("🚀 Jadio Interactive Text Generation")
        print("Type 'quit', 'exit', or 'q' to stop")
        print("Type 'help' for commands")
        print("-" * 50)
        
        # Default settings
        settings = {
            'max_new_tokens': 100,
            'temperature': 0.8,
            'top_k': 50,
            'top_p': 0.9,
            'do_sample': True,
            'num_return_sequences': 1
        }
        
        while True:
            try:
                user_input = input("\n📝 Enter prompt (or command): ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                elif user_input.lower() == 'help':
                    print("\n🔧 Available commands:")
                    print("  help - Show this help")
                    print("  settings - Show current settings")
                    print("  set <param> <value> - Change setting")
                    print("  quit/exit/q - Exit generator")
                    print("\n⚙️ Available parameters:")
                    for key, value in settings.items():
                        print(f"  {key}: {value}")
                    continue
                
                elif user_input.lower() == 'settings':
                    print("\n⚙️ Current settings:")
                    for key, value in settings.items():
                        print(f"  {key}: {value}")
                    continue
                
                elif user_input.lower().startswith('set '):
                    parts = user_input.split()
                    if len(parts) >= 3:
                        param = parts[1]
                        try:
                            if param in ['max_new_tokens', 'top_k', 'num_return_sequences']:
                                value = int(parts[2])
                            elif param in ['temperature', 'top_p']:
                                value = float(parts[2])
                            elif param == 'do_sample':
                                value = parts[2].lower() in ['true', '1', 'yes']
                            else:
                                print(f"❌ Unknown parameter: {param}")
                                continue
                            
                            settings[param] = value
                            print(f"✅ Set {param} = {value}")
                        except ValueError:
                            print(f"❌ Invalid value for {param}: {parts[2]}")
                    else:
                        print("❌ Usage: set <parameter> <value>")
                    continue
                
                elif not user_input:
                    print("❌ Please enter a prompt or command")
                    continue
                
                # Generate text
                print(f"\n🤖 Generating with: temp={settings['temperature']}, top_k={settings['top_k']}, top_p={settings['top_p']}")
                start_time = time.time()
                
                generated_texts = self.generate(
                    prompt=user_input,
                    **settings
                )
                
                generation_time = time.time() - start_time
                
                # Display results
                print(f"\n⚡ Generated in {generation_time:.2f}s")
                print("=" * 50)
                
                for i, text in enumerate(generated_texts):
                    if len(generated_texts) > 1:
                        print(f"\n🎯 Sequence {i+1}:")
                    print(f"💬 {text}")
                
                print("=" * 50)
                
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error during generation: {e}")
    
    def batch_generate(self, 
                      prompts: List[str],
                      output_file: Optional[str] = None,
                      **generation_kwargs) -> List[str]:
        """
        Generate text for a batch of prompts.
        
        Args:
            prompts: List of input prompts
            output_file: Optional file to save results
            **generation_kwargs: Arguments for generation
            
        Returns:
            List of generated texts
        """
        all_generated = []
        
        print(f"🚀 Generating for {len(prompts)} prompts...")
        
        for i, prompt in enumerate(prompts):
            print(f"Processing prompt {i+1}/{len(prompts)}: {prompt[:50]}...")
            
            generated = self.generate(prompt, **generation_kwargs)
            all_generated.extend(generated)
        
        # Save to file if specified
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                for i, (prompt, generated) in enumerate(zip(prompts, all_generated)):
                    f.write(f"=== Prompt {i+1} ===\n")
                    f.write(f"Input: {prompt}\n")
                    f.write(f"Output: {generated}\n\n")
            
            print(f"💾 Results saved to {output_file}")
        
        return all_generated


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Generate text with Jadio LLM")
    parser.add_argument("model_path", help="Path to model checkpoint")
    parser.add_argument("--tokenizer_path", help="Path to tokenizer files")
    parser.add_argument("--prompt", help="Text prompt for generation")
    parser.add_argument("--prompts_file", help="File containing prompts (one per line)")
    parser.add_argument("--output_file", help="File to save generated text")
    parser.add_argument("--max_new_tokens", type=int, default=100,
                       help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                       help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50,
                       help="Top-k sampling")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="Top-p (nucleus) sampling")
    parser.add_argument("--num_return_sequences", type=int, default=1,
                       help="Number of sequences to generate")
    parser.add_argument("--no_sample", action="store_true",
                       help="Use greedy decoding instead of sampling")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--device", choices=['auto', 'cuda', 'cpu'], default='auto',
                       help="Device to use")
    parser.add_argument("--interactive", action="store_true",
                       help="Start interactive generation mode")
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Initialize generator
    try:
        generator = JadioGenerator(
            model_path=args.model_path,
            tokenizer_path=args.tokenizer_path,
            device=args.device
        )
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return 1
    
    # Interactive mode
    if args.interactive:
        generator.interactive_generation()
        return 0
    
    # Prepare generation kwargs
    generation_kwargs = {
        'max_new_tokens': args.max_new_tokens,
        'temperature': args.temperature,
        'top_k': args.top_k,
        'top_p': args.top_p,
        'do_sample': not args.no_sample,
        'num_return_sequences': args.num_return_sequences
    }
    
    # Single prompt
    if args.prompt:
        print(f"🚀 Generating from prompt: {args.prompt}")
        
        start_time = time.time()
        generated_texts = generator.generate(args.prompt, **generation_kwargs)
        generation_time = time.time() - start_time
        
        print(f"\n⚡ Generated in {generation_time:.2f}s")
        print("=" * 50)
        
        for i, text in enumerate(generated_texts):
            if len(generated_texts) > 1:
                print(f"\n🎯 Sequence {i+1}:")
            print(f"💬 {text}")
        
        # Save if output file specified
        if args.output_file:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                f.write(f"Prompt: {args.prompt}\n\n")
                for i, text in enumerate(generated_texts):
                    if len(generated_texts) > 1:
                        f.write(f"=== Sequence {i+1} ===\n")
                    f.write(text + "\n\n")
            print(f"💾 Output saved to {args.output_file}")
    
    # Multiple prompts from file
    elif args.prompts_file:
        try:
            with open(args.prompts_file, 'r', encoding='utf-8') as f:
                prompts = [line.strip() for line in f if line.strip()]
            
            print(f"📁 Loaded {len(prompts)} prompts from {args.prompts_file}")
            
            generated_texts = generator.batch_generate(
                prompts,
                output_file=args.output_file,
                **generation_kwargs
            )
            
            print("✅ Batch generation completed!")
            
        except Exception as e:
            print(f"❌ Error processing prompts file: {e}")
            return 1
    
    else:
        print("❌ Please provide either --prompt, --prompts_file, or --interactive")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
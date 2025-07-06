"""
Jadio Tokenizer - GPT-2 compatible tokenizer for the Jadio LLM.
"""
import json
import os
import regex as re
from typing import List, Dict, Optional, Union, Tuple
from pathlib import Path
import torch


class JadioTokenizer:
    """
    GPT-2 compatible tokenizer for Jadio LLM.
    
    This tokenizer uses the same BPE (Byte-Pair Encoding) approach as GPT-2,
    ensuring compatibility with existing pretrained tokenizers.
    """
    
    def __init__(self, 
                 vocab_file: str = None,
                 merges_file: str = None,
                 tokenizer_config_file: str = None,
                 pad_token: str = "<|endoftext|>",
                 eos_token: str = "<|endoftext|>",
                 bos_token: str = "<|endoftext|>",
                 unk_token: str = "<|endoftext|>"):
        """
        Initialize the Jadio tokenizer.
        
        Args:
            vocab_file: Path to vocabulary JSON file
            merges_file: Path to merges.txt file
            tokenizer_config_file: Path to tokenizer config JSON
            pad_token: Padding token
            eos_token: End of sequence token
            bos_token: Beginning of sequence token
            unk_token: Unknown token
        """
        # Set default paths if not provided
        if vocab_file is None:
            current_dir = Path(__file__).parent
            vocab_file = current_dir / "vocab.json"
        if merges_file is None:
            current_dir = Path(__file__).parent
            merges_file = current_dir / "merges.txt"
        if tokenizer_config_file is None:
            current_dir = Path(__file__).parent
            tokenizer_config_file = current_dir / "tokenizer_config.json"
        
        # Load vocabulary
        self.vocab_file = vocab_file
        self.merges_file = merges_file
        
        # Load vocab
        with open(vocab_file, 'r', encoding='utf-8') as f:
            self.encoder = json.load(f)
        
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.vocab_size = len(self.encoder)
        
        # Load merges
        with open(merges_file, 'r', encoding='utf-8') as f:
            merges = f.read().strip().split('\n')[1:]  # Skip the first line
        
        self.bpe_ranks = dict(zip([tuple(merge.split()) for merge in merges], range(len(merges))))
        
        # Special tokens
        self.pad_token = pad_token
        self.eos_token = eos_token
        self.bos_token = bos_token
        self.unk_token = unk_token
        
        # Get token IDs
        self.pad_token_id = self.encoder.get(pad_token, 50256)
        self.eos_token_id = self.encoder.get(eos_token, 50256)
        self.bos_token_id = self.encoder.get(bos_token, 50256)
        self.unk_token_id = self.encoder.get(unk_token, 50256)
        
        # Regex pattern for tokenization (GPT-2 style)
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        
        # Byte encoder/decoder for handling arbitrary bytes
        self.byte_encoder = self._bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        
        # Cache for BPE
        self.cache = {}
    
    def _bytes_to_unicode(self):
        """
        Create a mapping from bytes to unicode characters.
        
        This is needed because GPT-2 tokenizer works with unicode strings,
        but we need to handle arbitrary bytes.
        """
        bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("¡"), ord("¬")+1)) + list(range(ord("®"), ord("ÿ")+1))
        cs = bs[:]
        n = 0
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8+n)
                n += 1
        cs = [chr(n) for n in cs]
        return dict(zip(bs, cs))
    
    def _get_pairs(self, word):
        """Get all pairs of consecutive tokens in a word."""
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs
    
    def _bpe(self, token):
        """
        Apply Byte-Pair Encoding to a token.
        
        Args:
            token: String token to encode
            
        Returns:
            Space-separated BPE tokens
        """
        if token in self.cache:
            return self.cache[token]
        
        word = tuple(token)
        pairs = self._get_pairs(word)
        
        if not pairs:
            return token
        
        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break
                
                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = self._get_pairs(word)
        
        word = ' '.join(word)
        self.cache[token] = word
        return word
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text string
            add_special_tokens: Whether to add BOS/EOS tokens
            
        Returns:
            List of token IDs
        """
        # Convert to bytes and then to unicode
        text_bytes = text.encode('utf-8')
        text = ''.join([self.byte_encoder[b] for b in text_bytes])
        
        # Apply regex pattern to split text
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            # Apply BPE
            bpe_token = self._bpe(token)
            bpe_tokens.extend([self.encoder[bpe_token] for bpe_token in bpe_token.split(' ')])
        
        # Add special tokens
        if add_special_tokens:
            # Note: GPT-2 style typically only adds EOS, not BOS
            bpe_tokens = bpe_tokens + [self.eos_token_id]
        
        return bpe_tokens
    
    def decode(self, token_ids: Union[List[int], torch.Tensor], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs or tensor
            skip_special_tokens: Whether to skip special tokens in output
            
        Returns:
            Decoded text string
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        # Convert IDs to tokens
        tokens = []
        for token_id in token_ids:
            if skip_special_tokens and token_id in [self.pad_token_id, self.eos_token_id, self.bos_token_id]:
                continue
            tokens.append(self.decoder.get(token_id, self.unk_token))
        
        # Join tokens and convert back to bytes
        text = ''.join(tokens)
        text_bytes = bytearray([self.byte_decoder[c] for c in text])
        
        # Decode to UTF-8
        try:
            return text_bytes.decode('utf-8')
        except UnicodeDecodeError:
            # Fallback: return the raw string
            return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into string tokens.
        
        Args:
            text: Input text string
            
        Returns:
            List of string tokens
        """
        token_ids = self.encode(text, add_special_tokens=False)
        return [self.decoder[token_id] for token_id in token_ids]
    
    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Convert string tokens to IDs."""
        return [self.encoder.get(token, self.unk_token_id) for token in tokens]
    
    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        """Convert token IDs to string tokens."""
        return [self.decoder.get(id, self.unk_token) for id in ids]
    
    def __len__(self):
        """Return vocabulary size."""
        return self.vocab_size
    
    def __call__(self, text: Union[str, List[str]], 
                 padding: Optional[Union[bool, str]] = None,
                 truncation: Optional[bool] = None,
                 max_length: Optional[int] = None,
                 return_tensors: Optional[str] = None,
                 add_special_tokens: bool = True) -> Dict[str, Union[List[int], torch.Tensor]]:
        """
        Tokenize and encode text with optional padding and truncation.
        
        Args:
            text: Input text(s)
            padding: Padding strategy ('max_length', 'longest', True, False)
            truncation: Whether to truncate sequences
            max_length: Maximum sequence length
            return_tensors: Return format ('pt' for PyTorch tensors)
            add_special_tokens: Whether to add special tokens
            
        Returns:
            Dictionary with 'input_ids' and 'attention_mask'
        """
        # Handle single string vs list of strings
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text
        
        # Encode all texts
        all_input_ids = []
        for t in texts:
            input_ids = self.encode(t, add_special_tokens=add_special_tokens)
            all_input_ids.append(input_ids)
        
        # Apply truncation
        if truncation and max_length:
            all_input_ids = [ids[:max_length] for ids in all_input_ids]
        
        # Apply padding
        if padding:
            if max_length is None:
                max_length = max(len(ids) for ids in all_input_ids)
            
            padded_input_ids = []
            attention_masks = []
            
            for input_ids in all_input_ids:
                # Pad sequence
                padding_length = max_length - len(input_ids)
                padded_ids = input_ids + [self.pad_token_id] * padding_length
                
                # Create attention mask
                attention_mask = [1] * len(input_ids) + [0] * padding_length
                
                padded_input_ids.append(padded_ids)
                attention_masks.append(attention_mask)
            
            all_input_ids = padded_input_ids
        else:
            attention_masks = [[1] * len(ids) for ids in all_input_ids]
        
        # Convert to tensors if requested
        result = {
            'input_ids': all_input_ids,
            'attention_mask': attention_masks
        }
        
        if return_tensors == 'pt':
            result['input_ids'] = torch.tensor(result['input_ids'], dtype=torch.long)
            result['attention_mask'] = torch.tensor(result['attention_mask'], dtype=torch.long)
        
        # Return single example if input was single string
        if isinstance(text, str):
            if return_tensors == 'pt':
                result['input_ids'] = result['input_ids'][0]
                result['attention_mask'] = result['attention_mask'][0]
            else:
                result['input_ids'] = result['input_ids'][0]
                result['attention_mask'] = result['attention_mask'][0]
        
        return result
    
    def save_pretrained(self, save_directory: str):
        """
        Save tokenizer files to directory.
        
        Args:
            save_directory: Directory to save tokenizer files
        """
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save vocabulary
        vocab_file = save_dir / "vocab.json"
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(self.encoder, f, ensure_ascii=False, indent=2)
        
        # Save merges
        merges_file = save_dir / "merges.txt"
        with open(merges_file, 'w', encoding='utf-8') as f:
            f.write("#version: 0.2\n")
            for bpe_tokens, _ in sorted(self.bpe_ranks.items(), key=lambda x: x[1]):
                f.write(' '.join(bpe_tokens) + '\n')
        
        # Save tokenizer config
        config_file = save_dir / "tokenizer_config.json"
        config = {
            "tokenizer_class": "JadioTokenizer",
            "pad_token": self.pad_token,
            "eos_token": self.eos_token,
            "bos_token": self.bos_token,
            "unk_token": self.unk_token,
            "vocab_size": self.vocab_size,
        }
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def from_pretrained(cls, model_path: str):
        """
        Load tokenizer from pretrained files.
        
        Args:
            model_path: Path to directory containing tokenizer files
            
        Returns:
            JadioTokenizer instance
        """
        model_dir = Path(model_path)
        
        vocab_file = model_dir / "vocab.json"
        merges_file = model_dir / "merges.txt"
        config_file = model_dir / "tokenizer_config.json"
        
        # Load config if available
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            return cls(
                vocab_file=str(vocab_file),
                merges_file=str(merges_file),
                tokenizer_config_file=str(config_file),
                pad_token=config.get("pad_token", "<|endoftext|>"),
                eos_token=config.get("eos_token", "<|endoftext|>"),
                bos_token=config.get("bos_token", "<|endoftext|>"),
                unk_token=config.get("unk_token", "<|endoftext|>")
            )
        else:
            return cls(
                vocab_file=str(vocab_file),
                merges_file=str(merges_file)
            )


def main():
    """Test the tokenizer."""
    print("[Jadio] Testing Tokenizer...")
    
    # Try to load from the current directory (where GPT-2 files should be)
    current_dir = Path(__file__).parent
    
    # Check if files exist
    vocab_file = current_dir / "vocab.json"
    merges_file = current_dir / "merges.txt"
    
    if not vocab_file.exists() or not merges_file.exists():
        print(f"Warning: GPT-2 tokenizer files not found in {current_dir}")
        print("Expected files: vocab.json, merges.txt")
        print("Please ensure these files are present to test the tokenizer.")
        return
    
    try:
        # Initialize tokenizer
        tokenizer = JadioTokenizer(
            vocab_file=str(vocab_file),
            merges_file=str(merges_file)
        )
        
        print(f"Tokenizer loaded successfully!")
        print(f"Vocabulary size: {len(tokenizer)}")
        print(f"Pad token ID: {tokenizer.pad_token_id}")
        print(f"EOS token ID: {tokenizer.eos_token_id}")
        
        # Test encoding/decoding
        test_texts = [
            "Hello, world!",
            "The quick brown fox jumps over the lazy dog.",
            "This is a test of the Jadio tokenizer with multiple sentences. How well does it work?",
            "😀 Emojis and unicode characters! 🚀",
            "Numbers: 123, 456.789, and special chars: @#$%^&*()"
        ]
        
        print("\n--- Testing Encoding/Decoding ---")
        for text in test_texts:
            # Encode
            token_ids = tokenizer.encode(text)
            tokens = tokenizer.tokenize(text)
            
            # Decode
            decoded = tokenizer.decode(token_ids)
            
            print(f"\nOriginal: {repr(text)}")
            print(f"Tokens ({len(tokens)}): {tokens[:10]}{'...' if len(tokens) > 10 else ''}")
            print(f"Token IDs ({len(token_ids)}): {token_ids[:10]}{'...' if len(token_ids) > 10 else ''}")
            print(f"Decoded: {repr(decoded)}")
            print(f"Round-trip success: {text == decoded}")
        
        # Test batch processing
        print("\n--- Testing Batch Processing ---")
        batch_texts = test_texts[:3]
        
        # Without padding
        batch_encoded = tokenizer(batch_texts, add_special_tokens=True)
        print(f"Batch encoding (no padding):")
        for i, (text, ids) in enumerate(zip(batch_texts, batch_encoded['input_ids'])):
            print(f"  Text {i}: {len(ids)} tokens")
        
        # With padding
        batch_encoded_padded = tokenizer(
            batch_texts, 
            padding=True, 
            max_length=50,
            return_tensors='pt'
        )
        print(f"\nBatch encoding (with padding):")
        print(f"Input IDs shape: {batch_encoded_padded['input_ids'].shape}")
        print(f"Attention mask shape: {batch_encoded_padded['attention_mask'].shape}")
        
        # Test truncation
        long_text = "This is a very long text that should be truncated. " * 20
        truncated = tokenizer(
            long_text,
            max_length=50,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        print(f"\nTruncation test:")
        print(f"Original length: ~{len(tokenizer.encode(long_text))} tokens")
        print(f"Truncated shape: {truncated['input_ids'].shape}")
        
        print("\n[Jadio] Tokenizer test completed successfully!")
        
    except Exception as e:
        print(f"Error testing tokenizer: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
"""
Dataset loader for the Jadio LLM.
"""
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from typing import List, Dict, Optional, Union, Iterator, Any
import random
from pathlib import Path
import mmap
import gzip
from tqdm import tqdm

# Import our tokenizer
from ..tokenizer.jadio_tokenizer import JadioTokenizer


class TextDataset(Dataset):
    """
    Dataset for text data with tokenization.
    
    This dataset loads text data and tokenizes it on-the-fly.
    Suitable for smaller datasets that fit in memory.
    """
    
    def __init__(self, 
                 texts: List[str],
                 tokenizer: JadioTokenizer,
                 max_length: int = 1024,
                 return_tensors: bool = True):
        """
        Initialize text dataset.
        
        Args:
            texts: List of text strings
            tokenizer: Jadio tokenizer instance
            max_length: Maximum sequence length
            return_tensors: Whether to return PyTorch tensors
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.return_tensors = return_tensors
        
        # Pre-tokenize for efficiency (optional)
        self.tokenized_texts = []
        print("Pre-tokenizing texts...")
        for text in tqdm(texts):
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            # Truncate if necessary
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            self.tokenized_texts.append(tokens)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        tokens = self.tokenized_texts[idx]
        
        if self.return_tensors:
            return {
                'input_ids': torch.tensor(tokens, dtype=torch.long),
                'attention_mask': torch.ones(len(tokens), dtype=torch.long)
            }
        else:
            return {
                'input_ids': tokens,
                'attention_mask': [1] * len(tokens)
            }


class StreamingTextDataset(IterableDataset):
    """
    Streaming dataset for large text files.
    
    This dataset reads text data line-by-line from files without loading
    everything into memory. Suitable for very large datasets.
    """
    
    def __init__(self,
                 file_paths: Union[str, List[str]],
                 tokenizer: JadioTokenizer,
                 max_length: int = 1024,
                 buffer_size: int = 10000,
                 shuffle_buffer: bool = True):
        """
        Initialize streaming text dataset.
        
        Args:
            file_paths: Path(s) to text files
            tokenizer: Jadio tokenizer instance
            max_length: Maximum sequence length
            buffer_size: Size of shuffle buffer
            shuffle_buffer: Whether to shuffle data in buffer
        """
        if isinstance(file_paths, str):
            file_paths = [file_paths]
        
        self.file_paths = file_paths
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.buffer_size = buffer_size
        self.shuffle_buffer = shuffle_buffer
    
    def _read_files(self) -> Iterator[str]:
        """Read lines from all files."""
        for file_path in self.file_paths:
            if file_path.endswith('.gz'):
                open_fn = gzip.open
                mode = 'rt'
            else:
                open_fn = open
                mode = 'r'
            
            with open_fn(file_path, mode, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:  # Skip empty lines
                        yield line
    
    def _process_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Process text into model inputs."""
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        
        # Truncate if necessary
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        
        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'attention_mask': torch.ones(len(tokens), dtype=torch.long)
        }
    
    def __iter__(self):
        """Iterate over dataset."""
        if self.shuffle_buffer:
            # Use shuffle buffer
            buffer = []
            for text in self._read_files():
                buffer.append(text)
                
                if len(buffer) >= self.buffer_size:
                    random.shuffle(buffer)
                    for item in buffer:
                        yield self._process_text(item)
                    buffer = []
            
            # Process remaining items
            if buffer:
                random.shuffle(buffer)
                for item in buffer:
                    yield self._process_text(item)
        else:
            # No shuffling
            for text in self._read_files():
                yield self._process_text(text)


class ConcatenatedDataset(Dataset):
    """
    Dataset that concatenates tokenized texts into fixed-length sequences.
    
    This is useful for training language models where you want to pack
    multiple texts into sequences of a fixed length for efficiency.
    """
    
    def __init__(self,
                 texts: List[str],
                 tokenizer: JadioTokenizer,
                 max_length: int = 1024,
                 stride: Optional[int] = None):
        """
        Initialize concatenated dataset.
        
        Args:
            texts: List of text strings
            tokenizer: Jadio tokenizer instance
            max_length: Maximum sequence length
            stride: Stride for overlapping sequences (default: max_length)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride or max_length
        
        # Tokenize and concatenate all texts
        print("Tokenizing and concatenating texts...")
        all_tokens = []
        for text in tqdm(texts):
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            all_tokens.extend(tokens)
            # Add EOS token between documents
            all_tokens.append(self.tokenizer.eos_token_id)
        
        # Split into sequences
        self.sequences = []
        for i in range(0, len(all_tokens) - max_length + 1, self.stride):
            sequence = all_tokens[i:i + max_length]
            self.sequences.append(sequence)
        
        print(f"Created {len(self.sequences)} sequences of length {max_length}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        tokens = self.sequences[idx]
        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'attention_mask': torch.ones(len(tokens), dtype=torch.long)
        }


class JsonlDataset(Dataset):
    """
    Dataset for JSONL (JSON Lines) format data.
    
    Each line in the file should be a JSON object with a 'text' field.
    """
    
    def __init__(self,
                 file_path: str,
                 tokenizer: JadioTokenizer,
                 max_length: int = 1024,
                 text_field: str = 'text'):
        """
        Initialize JSONL dataset.
        
        Args:
            file_path: Path to JSONL file
            tokenizer: Jadio tokenizer instance
            max_length: Maximum sequence length
            text_field: Name of the text field in JSON objects
        """
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_field = text_field
        
        # Count lines for __len__
        with open(file_path, 'r', encoding='utf-8') as f:
            self.num_lines = sum(1 for _ in f)
        
        print(f"Loaded JSONL dataset with {self.num_lines} examples")
    
    def __len__(self):
        return self.num_lines
    
    def __getitem__(self, idx):
        # Read specific line (not very efficient for random access)
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i == idx:
                    data = json.loads(line.strip())
                    text = data[self.text_field]
                    
                    tokens = self.tokenizer.encode(text, add_special_tokens=True)
                    if len(tokens) > self.max_length:
                        tokens = tokens[:self.max_length]
                    
                    return {
                        'input_ids': torch.tensor(tokens, dtype=torch.long),
                        'attention_mask': torch.ones(len(tokens), dtype=torch.long)
                    }
        
        raise IndexError(f"Index {idx} out of range")


def collate_fn(batch: List[Dict[str, torch.Tensor]], 
               pad_token_id: int = 50256) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader.
    
    Pads sequences to the same length within a batch.
    
    Args:
        batch: List of samples from dataset
        pad_token_id: Token ID to use for padding
        
    Returns:
        Batched and padded tensors
    """
    # Get maximum length in batch
    max_length = max(len(sample['input_ids']) for sample in batch)
    
    # Pad all sequences
    input_ids = []
    attention_masks = []
    
    for sample in batch:
        ids = sample['input_ids']
        mask = sample['attention_mask']
        
        # Calculate padding needed
        padding_length = max_length - len(ids)
        
        # Pad input_ids
        padded_ids = torch.cat([
            ids, 
            torch.full((padding_length,), pad_token_id, dtype=torch.long)
        ])
        
        # Pad attention_mask
        padded_mask = torch.cat([
            mask,
            torch.zeros(padding_length, dtype=torch.long)
        ])
        
        input_ids.append(padded_ids)
        attention_masks.append(padded_mask)
    
    return {
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attention_masks)
    }


def create_dataloader(dataset: Dataset,
                     batch_size: int = 8,
                     shuffle: bool = True,
                     num_workers: int = 0,
                     pad_token_id: int = 50256) -> DataLoader:
    """
    Create a DataLoader for training.
    
    Args:
        dataset: Dataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pad_token_id: Padding token ID
        
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_fn(batch, pad_token_id),
        pin_memory=torch.cuda.is_available()
    )


def load_text_files(directory: str, extensions: List[str] = ['.txt', '.json']) -> List[str]:
    """
    Load text from all files in a directory.
    
    Args:
        directory: Path to directory containing text files
        extensions: File extensions to include
        
    Returns:
        List of text strings
    """
    texts = []
    directory = Path(directory)
    
    for ext in extensions:
        for file_path in directory.glob(f'*{ext}'):
            print(f"Loading {file_path}")
            try:
                if ext == '.json':
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            for item in data:
                                if isinstance(item, dict) and 'text' in item:
                                    texts.append(item['text'])
                                elif isinstance(item, str):
                                    texts.append(item)
                        elif isinstance(data, dict) and 'text' in data:
                            texts.append(data['text'])
                        elif isinstance(data, str):
                            texts.append(data)
                else:  # .txt files
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content:
                            texts.append(content)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    
    print(f"Loaded {len(texts)} texts from {directory}")
    return texts


def create_dummy_texts(num_texts: int = 1000, min_length: int = 50, max_length: int = 500) -> List[str]:
    """
    Create dummy text data for testing.
    
    Args:
        num_texts: Number of texts to generate
        min_length: Minimum text length (in characters)
        max_length: Maximum text length (in characters)
        
    Returns:
        List of dummy text strings
    """
    import random
    import string
    
    # Sample words for generating text
    words = [
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "and", "runs",
        "through", "forest", "with", "great", "speed", "while", "birds", "sing", "in",
        "trees", "above", "ground", "where", "flowers", "bloom", "during", "spring",
        "time", "when", "nature", "comes", "alive", "again", "after", "long", "winter",
        "months", "that", "seem", "endless", "sometimes", "but", "always", "give", "way",
        "to", "warmer", "days", "filled", "sunshine", "happiness", "joy", "laughter"
    ]
    
    texts = []
    for _ in range(num_texts):
        # Random text length
        target_length = random.randint(min_length, max_length)
        
        # Generate text
        text_words = []
        current_length = 0
        
        while current_length < target_length:
            word = random.choice(words)
            text_words.append(word)
            current_length += len(word) + 1  # +1 for space
        
        text = ' '.join(text_words)
        
        # Add some punctuation
        text = text.replace(' and ', ', and ')
        text = text.replace(' but ', ', but ')
        text = text.replace(' while ', ', while ')
        text += '.'
        
        # Capitalize first letter
        text = text[0].upper() + text[1:]
        
        texts.append(text)
    
    return texts


def main():
    """Test the dataset loader."""
    print("[Jadio] Testing Dataset Loader...")
    
    # Try to load tokenizer
    try:
        from ..tokenizer.jadio_tokenizer import JadioTokenizer
        
        # Check if tokenizer files exist
        current_dir = Path(__file__).parent.parent / "modelling" / "jadio01"
        vocab_file = current_dir / "vocab.json"
        
        if vocab_file.exists():
            tokenizer = JadioTokenizer.from_pretrained(str(current_dir))
            print(f"Loaded tokenizer with vocab size: {len(tokenizer)}")
        else:
            print("Tokenizer files not found, creating dummy tokenizer for testing")
            # Create a minimal dummy tokenizer for testing
            tokenizer = type('DummyTokenizer', (), {
                'encode': lambda self, text, add_special_tokens=True: list(range(min(len(text), 50))),
                'decode': lambda self, tokens, skip_special_tokens=True: ' '.join(map(str, tokens)),
                'eos_token_id': 0,
                'pad_token_id': 0,
                '__len__': lambda self: 1000
            })()
    
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return
    
    # Create dummy data
    print("\n--- Creating dummy data ---")
    dummy_texts = create_dummy_texts(100, 50, 200)
    print(f"Created {len(dummy_texts)} dummy texts")
    print(f"Sample text: {dummy_texts[0][:100]}...")
    
    # Test TextDataset
    print("\n--- Testing TextDataset ---")
    text_dataset = TextDataset(
        texts=dummy_texts[:50],
        tokenizer=tokenizer,
        max_length=128
    )
    
    print(f"Dataset size: {len(text_dataset)}")
    sample = text_dataset[0]
    print(f"Sample input_ids shape: {sample['input_ids'].shape}")
    print(f"Sample attention_mask shape: {sample['attention_mask'].shape}")
    
    # Test DataLoader
    print("\n--- Testing DataLoader ---")
    dataloader = create_dataloader(
        text_dataset,
        batch_size=4,
        shuffle=True
    )
    
    batch = next(iter(dataloader))
    print(f"Batch input_ids shape: {batch['input_ids'].shape}")
    print(f"Batch attention_mask shape: {batch['attention_mask'].shape}")
    
    # Test ConcatenatedDataset
    print("\n--- Testing ConcatenatedDataset ---")
    concat_dataset = ConcatenatedDataset(
        texts=dummy_texts[:20],
        tokenizer=tokenizer,
        max_length=64,
        stride=32
    )
    
    print(f"Concatenated dataset size: {len(concat_dataset)}")
    sample = concat_dataset[0]
    print(f"Sample shape: {sample['input_ids'].shape}")
    
    print("\n[Jadio] Dataset loader test completed!")


if __name__ == "__main__":
    main()
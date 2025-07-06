"""
Evaluation metrics for the Jadio LLM.
"""
import torch
import torch.nn.functional as F
import numpy as np
import math
from typing import List, Dict, Any, Optional, Union, Tuple
from collections import Counter
import re


def perplexity(loss: Union[float, torch.Tensor]) -> float:
    """
    Calculate perplexity from cross-entropy loss.
    
    Args:
        loss: Cross-entropy loss
        
    Returns:
        Perplexity value
    """
    if isinstance(loss, torch.Tensor):
        loss = loss.item()
    return math.exp(loss)


def accuracy(logits: torch.Tensor, 
             targets: torch.Tensor, 
             ignore_index: int = -100) -> float:
    """
    Calculate token-level accuracy.
    
    Args:
        logits: Model predictions (batch_size, seq_len, vocab_size)
        targets: Target token IDs (batch_size, seq_len)
        ignore_index: Index to ignore in accuracy calculation
        
    Returns:
        Accuracy as a float between 0 and 1
    """
    predictions = torch.argmax(logits, dim=-1)
    
    if ignore_index is not None:
        mask = (targets != ignore_index)
        correct = (predictions == targets) & mask
        total = mask.sum().item()
    else:
        correct = (predictions == targets)
        total = targets.numel()
    
    if total == 0:
        return 0.0
    
    return correct.sum().item() / total


def top_k_accuracy(logits: torch.Tensor, 
                   targets: torch.Tensor, 
                   k: int = 5,
                   ignore_index: int = -100) -> float:
    """
    Calculate top-k accuracy.
    
    Args:
        logits: Model predictions (batch_size, seq_len, vocab_size)
        targets: Target token IDs (batch_size, seq_len)
        k: Number of top predictions to consider
        ignore_index: Index to ignore in accuracy calculation
        
    Returns:
        Top-k accuracy as a float between 0 and 1
    """
    _, top_k_preds = torch.topk(logits, k, dim=-1)
    targets_expanded = targets.unsqueeze(-1).expand(-1, -1, k)
    
    if ignore_index is not None:
        mask = (targets != ignore_index)
        correct = (top_k_preds == targets_expanded).any(dim=-1) & mask
        total = mask.sum().item()
    else:
        correct = (top_k_preds == targets_expanded).any(dim=-1)
        total = targets.numel()
    
    if total == 0:
        return 0.0
    
    return correct.sum().item() / total


def bits_per_byte(loss: Union[float, torch.Tensor]) -> float:
    """
    Calculate bits per byte from cross-entropy loss.
    
    This is a common metric for character-level language models.
    
    Args:
        loss: Cross-entropy loss (nats)
        
    Returns:
        Bits per byte
    """
    if isinstance(loss, torch.Tensor):
        loss = loss.item()
    
    # Convert from nats to bits, then normalize by log(256) for bytes
    return loss / math.log(2) / math.log(256)


def sequence_accuracy(logits: torch.Tensor, 
                     targets: torch.Tensor,
                     ignore_index: int = -100) -> float:
    """
    Calculate sequence-level accuracy (all tokens must be correct).
    
    Args:
        logits: Model predictions (batch_size, seq_len, vocab_size)
        targets: Target token IDs (batch_size, seq_len)
        ignore_index: Index to ignore in accuracy calculation
        
    Returns:
        Sequence accuracy as a float between 0 and 1
    """
    predictions = torch.argmax(logits, dim=-1)
    
    if ignore_index is not None:
        mask = (targets != ignore_index)
        correct_tokens = (predictions == targets) | ~mask
    else:
        correct_tokens = (predictions == targets)
    
    # All tokens in sequence must be correct
    correct_sequences = correct_tokens.all(dim=-1)
    
    return correct_sequences.float().mean().item()


def bleu_score(predictions: List[str], 
               references: List[str], 
               n_grams: int = 4) -> Dict[str, float]:
    """
    Calculate BLEU score for generated text.
    
    This is a simplified BLEU implementation. For production use,
    consider using nltk.translate.bleu_score or sacrebleu.
    
    Args:
        predictions: List of generated texts
        references: List of reference texts
        n_grams: Maximum n-gram order to consider
        
    Returns:
        Dictionary with BLEU scores
    """
    def tokenize(text: str) -> List[str]:
        """Simple tokenization."""
        return re.findall(r'\w+', text.lower())
    
    def get_ngrams(tokens: List[str], n: int) -> Counter:
        """Get n-grams from tokens."""
        return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))
    
    def modified_precision(pred_tokens: List[str], 
                          ref_tokens: List[str], 
                          n: int) -> float:
        """Calculate modified precision for n-grams."""
        pred_ngrams = get_ngrams(pred_tokens, n)
        ref_ngrams = get_ngrams(ref_tokens, n)
        
        if not pred_ngrams:
            return 0.0
        
        numerator = sum(min(count, ref_ngrams[ngram]) 
                       for ngram, count in pred_ngrams.items())
        denominator = sum(pred_ngrams.values())
        
        return numerator / denominator if denominator > 0 else 0.0
    
    if len(predictions) != len(references):
        raise ValueError("Number of predictions must match number of references")
    
    total_precision = [0.0] * n_grams
    total_pred_length = 0
    total_ref_length = 0
    
    for pred, ref in zip(predictions, references):
        pred_tokens = tokenize(pred)
        ref_tokens = tokenize(ref)
        
        total_pred_length += len(pred_tokens)
        total_ref_length += len(ref_tokens)
        
        for n in range(1, n_grams + 1):
            precision = modified_precision(pred_tokens, ref_tokens, n)
            total_precision[n-1] += precision
    
    # Average precision scores
    avg_precision = [p / len(predictions) for p in total_precision]
    
    # Brevity penalty
    if total_pred_length == 0:
        bp = 0.0
    elif total_pred_length >= total_ref_length:
        bp = 1.0
    else:
        bp = math.exp(1 - total_ref_length / total_pred_length)
    
    # Calculate BLEU score
    if all(p > 0 for p in avg_precision):
        log_bleu = sum(math.log(p) for p in avg_precision) / n_grams
        bleu = bp * math.exp(log_bleu)
    else:
        bleu = 0.0
    
    return {
        'bleu': bleu,
        'brevity_penalty': bp,
        'precision_1': avg_precision[0] if n_grams >= 1 else 0.0,
        'precision_2': avg_precision[1] if n_grams >= 2 else 0.0,
        'precision_3': avg_precision[2] if n_grams >= 3 else 0.0,
        'precision_4': avg_precision[3] if n_grams >= 4 else 0.0,
    }


def rouge_l_score(predictions: List[str], 
                  references: List[str]) -> Dict[str, float]:
    """
    Calculate ROUGE-L score for generated text.
    
    ROUGE-L is based on the longest common subsequence.
    
    Args:
        predictions: List of generated texts
        references: List of reference texts
        
    Returns:
        Dictionary with ROUGE-L scores
    """
    def tokenize(text: str) -> List[str]:
        """Simple tokenization."""
        return re.findall(r'\w+', text.lower())
    
    def lcs_length(x: List[str], y: List[str]) -> int:
        """Calculate longest common subsequence length."""
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i-1] == y[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    if len(predictions) != len(references):
        raise ValueError("Number of predictions must match number of references")
    
    total_lcs = 0
    total_pred_length = 0
    total_ref_length = 0
    
    for pred, ref in zip(predictions, references):
        pred_tokens = tokenize(pred)
        ref_tokens = tokenize(ref)
        
        lcs_len = lcs_length(pred_tokens, ref_tokens)
        
        total_lcs += lcs_len
        total_pred_length += len(pred_tokens)
        total_ref_length += len(ref_tokens)
    
    # Calculate precision, recall, and F1
    precision = total_lcs / total_pred_length if total_pred_length > 0 else 0.0
    recall = total_lcs / total_ref_length if total_ref_length > 0 else 0.0
    
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    
    return {
        'rouge_l_precision': precision,
        'rouge_l_recall': recall,
        'rouge_l_f1': f1
    }


def diversity_metrics(texts: List[str]) -> Dict[str, float]:
    """
    Calculate diversity metrics for generated text.
    
    Args:
        texts: List of generated texts
        
    Returns:
        Dictionary with diversity metrics
    """
    def tokenize(text: str) -> List[str]:
        """Simple tokenization."""
        return re.findall(r'\w+', text.lower())
    
    all_tokens = []
    all_bigrams = []
    all_trigrams = []
    
    for text in texts:
        tokens = tokenize(text)
        all_tokens.extend(tokens)
        
        # Get bigrams and trigrams
        bigrams = [tuple(tokens[i:i+2]) for i in range(len(tokens) - 1)]
        trigrams = [tuple(tokens[i:i+3]) for i in range(len(tokens) - 2)]
        
        all_bigrams.extend(bigrams)
        all_trigrams.extend(trigrams)
    
    # Calculate distinct n-gram ratios
    distinct_1 = len(set(all_tokens)) / len(all_tokens) if all_tokens else 0.0
    distinct_2 = len(set(all_bigrams)) / len(all_bigrams) if all_bigrams else 0.0
    distinct_
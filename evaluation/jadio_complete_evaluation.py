"""
Evaluation module for the Jadio LLM.
"""
import torch
import torch.nn.functional as F
import numpy as np
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from tqdm import tqdm
from dataclasses import dataclass

from ..config.jadio_config import JadioConfig
from ..modelling.jadio01.jadio_decoder_transformer import JadioLMHeadModel
from ..tokenizer.jadio_tokenizer import JadioTokenizer
from ..data.jadio_dataset_loader import create_dataloader
from .jadio_metrics import MetricsCalculator
from ..scripts.jadio_utilities import setup_logger


@dataclass
class EvaluationResults:
    """Container for evaluation results."""
    model_name: str
    dataset_name: str
    metrics: Dict[str, float]
    generation_samples: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    timestamp: str


class JadioEvaluator:
    """
    Comprehensive evaluator for Jadio LLM models.
    
    Handles perplexity evaluation, generation quality assessment,
    and benchmark testing.
    """
    
    def __init__(self, 
                 model: JadioLMHeadModel,
                 tokenizer: JadioTokenizer,
                 device: str = 'auto'):
        """
        Initialize evaluator.
        
        Args:
            model: Jadio model to evaluate
            tokenizer: Tokenizer for text processing
            device: Device to use for evaluation
        """
        self.model = model
        self.tokenizer = tokenizer
        
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        self.model.eval()
        
        self.metrics_calculator = MetricsCalculator(tokenizer)
        self.logger = setup_logger("jadio_evaluator")
        
        self.logger.info(f"Initialized evaluator on {self.device}")
        self.logger.info(f"Model: {self.model.get_num_params() / 1e6:.1f}M parameters")
    
    def evaluate_perplexity(self,
                           dataloader,
                           max_steps: Optional[int] = None,
                           return_details: bool = False) -> Dict[str, float]:
        """
        Evaluate model perplexity on a dataset.
        
        Args:
            dataloader: DataLoader for evaluation data
            max_steps: Maximum number of evaluation steps
            return_details: Whether to return detailed metrics
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.logger.info("Starting perplexity evaluation...")
        
        total_loss = 0.0
        total_tokens = 0
        total_correct = 0
        batch_losses = []
        
        start_time = time.time()
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
                if max_steps and i >= max_steps:
                    break
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch.get('attention_mask', None)
                
                # Forward pass
                outputs = self.model(input_ids, labels=input_ids)
                loss = outputs[0] if isinstance(outputs, tuple) else outputs.loss
                logits = outputs[1] if isinstance(outputs, tuple) else outputs.logits
                
                # Calculate metrics
                batch_metrics = self.metrics_calculator.calculate_language_modeling_metrics(
                    logits, input_ids, loss
                )
                
                # Accumulate stats
                batch_size = input_ids.size(0)
                seq_len = input_ids.size(1)
                num_tokens = batch_size * seq_len
                
                total_loss += batch_metrics['loss'] * num_tokens
                total_tokens += num_tokens
                total_correct += batch_metrics['accuracy'] * num_tokens
                batch_losses.append(batch_metrics['loss'])
        
        eval_time = time.time() - start_time
        
        # Calculate final metrics
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        avg_accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
        perplexity = np.exp(avg_loss)
        
        results = {
            'loss': avg_loss,
            'perplexity': perplexity,
            'accuracy': avg_accuracy,
            'total_tokens': total_tokens,
            'eval_time': eval_time,
            'tokens_per_second': total_tokens / eval_time if eval_time > 0 else 0.0
        }
        
        if return_details:
            results.update({
                'loss_std': np.std(batch_losses),
                'loss_min': np.min(batch_losses),
                'loss_max': np.max(batch_losses),
                'num_batches': len(batch_losses)
            })
        
        self.logger.info(f"Perplexity evaluation completed:")
        self.logger.info(f"  Loss: {avg_loss:.4f}")
        self.logger.info(f"  Perplexity: {perplexity:.2f}")
        self.logger.info(f"  Accuracy: {avg_accuracy:.4f}")
        
        return results
    
    def evaluate_generation(self,
                           prompts: List[str],
                           generation_config: Optional[Dict[str, Any]] = None,
                           reference_texts: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Evaluate text generation quality.
        
        Args:
            prompts: List of prompts for generation
            generation_config: Generation parameters
            reference_texts: Reference texts for comparison
            
        Returns:
            Generation evaluation results
        """
        self.logger.info(f"Starting generation evaluation with {len(prompts)} prompts...")
        
        # Default generation config
        gen_config = {
            'max_new_tokens': 100,
            'temperature': 0.8,
            'top_k': 50,
            'top_p': 0.9,
            'do_sample': True,
            'num_return_sequences': 1
        }
        if generation_config:
            gen_config.update(generation_config)
        
        generated_texts = []
        generation_times = []
        
        # Generate text for each prompt
        for prompt in tqdm(prompts, desc="Generating"):
            start_time = time.time()
            
            # Tokenize prompt
            input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
            input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device).unsqueeze(0)
            
            # Generate
            with torch.no_grad():
                generated = self.model.generate(input_ids, **gen_config)
            
            # Decode
            generated_tokens = generated[0].cpu().tolist()
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Remove prompt from generated text
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            generated_texts.append(generated_text)
            generation_times.append(time.time() - start_time)
        
        # Calculate generation metrics
        generation_metrics = self.metrics_calculator.calculate_generation_metrics(
            generated_texts, reference_texts
        )
        
        # Add timing metrics
        generation_metrics.update({
            'avg_generation_time': np.mean(generation_times),
            'total_generation_time': np.sum(generation_times),
            'generations_per_second': len(prompts) / np.sum(generation_times)
        })
        
        # Create sample outputs
        samples = []
        for i, (prompt, generated, gen_time) in enumerate(zip(prompts, generated_texts, generation_times)):
            sample = {
                'prompt': prompt,
                'generated': generated,
                'generation_time': gen_time,
                'length': len(generated.split())
            }
            if reference_texts and i < len(reference_texts):
                sample['reference'] = reference_texts[i]
            samples.append(sample)
        
        results = {
            'metrics': generation_metrics,
            'samples': samples,
            'generation_config': gen_config,
            'num_prompts': len(prompts)
        }
        
        self.logger.info("Generation evaluation completed:")
        self.logger.info(f"  Average generation time: {generation_metrics['avg_generation_time']:.2f}s")
        self.logger.info(f"  Distinct-1: {generation_metrics['distinct_1']:.3f}")
        self.logger.info(f"  Distinct-2: {generation_metrics['distinct_2']:.3f}")
        
        return results
    
    def run_benchmark(self,
                     benchmark_name: str,
                     test_data: Dict[str, Any],
                     output_dir: str = "evaluation_results") -> EvaluationResults:
        """
        Run a comprehensive benchmark evaluation.
        
        Args:
            benchmark_name: Name of the benchmark
            test_data: Test data containing prompts and references
            output_dir: Directory to save results
            
        Returns:
            Complete evaluation results
        """
        self.logger.info(f"Starting benchmark: {benchmark_name}")
        
        results_dir = Path(output_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract test components
        prompts = test_data.get('prompts', [])
        references = test_data.get('references', None)
        eval_dataloader = test_data.get('dataloader', None)
        
        all_metrics = {}
        samples = []
        
        # Perplexity evaluation
        if eval_dataloader:
            self.logger.info("Running perplexity evaluation...")
            perplexity_results = self.evaluate_perplexity(eval_dataloader, max_steps=100)
            all_metrics.update({f"perplexity_{k}": v for k, v in perplexity_results.items()})
        
        # Generation evaluation
        if prompts:
            self.logger.info("Running generation evaluation...")
            generation_results = self.evaluate_generation(prompts, references=references)
            all_metrics.update({f"generation_{k}": v for k, v in generation_results['metrics'].items()})
            samples = generation_results['samples']
        
        # Create results object
        results = EvaluationResults(
            model_name=getattr(self.model, 'name', 'jadio_model'),
            dataset_name=benchmark_name,
            metrics=all_metrics,
            generation_samples=samples[:10],  # Save first 10 samples
            metadata={
                'model_parameters': self.model.get_num_params(),
                'device': str(self.device),
                'config': self.model.config.to_dict() if hasattr(self.model, 'config') else {},
                'tokenizer_vocab_size': len(self.tokenizer),
                'evaluation_date': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            timestamp=time.strftime('%Y%m%d_%H%M%S')
        )
        
        # Save results
        results_file = results_dir / f"{benchmark_name}_{results.timestamp}.json"
        self._save_results(results, results_file)
        
        self.logger.info(f"Benchmark completed. Results saved to {results_file}")
        return results
    
    def _save_results(self, results: EvaluationResults, filepath: Path) -> None:
        """Save evaluation results to file."""
        results_dict = {
            'model_name': results.model_name,
            'dataset_name': results.dataset_name,
            'metrics': results.metrics,
            'generation_samples': results.generation_samples,
            'metadata': results.metadata,
            'timestamp': results.timestamp
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
    
    def compare_models(self,
                      other_evaluators: List['JadioEvaluator'],
                      test_data: Dict[str, Any],
                      model_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Compare multiple models on the same test data.
        
        Args:
            other_evaluators: List of other evaluators
            test_data: Test data for comparison
            model_names: Names for the models
            
        Returns:
            Comparison results
        """
        evaluators = [self] + other_evaluators
        if model_names is None:
            model_names = [f"model_{i}" for i in range(len(evaluators))]
        
        self.logger.info(f"Comparing {len(evaluators)} models...")
        
        comparison_results = {
            'models': model_names,
            'individual_results': {},
            'comparisons': {}
        }
        
        # Evaluate each model
        for evaluator, name in zip(evaluators, model_names):
            self.logger.info(f"Evaluating {name}...")
            results = evaluator.run_benchmark(f"comparison_{name}", test_data)
            comparison_results['individual_results'][name] = results.metrics
        
        # Create comparisons
        metrics_to_compare = ['perplexity_perplexity', 'generation_distinct_1', 'generation_distinct_2']
        
        for metric in metrics_to_compare:
            if all(metric in results for results in comparison_results['individual_results'].values()):
                values = [comparison_results['individual_results'][name][metric] for name in model_names]
                comparison_results['comparisons'][metric] = {
                    'values': dict(zip(model_names, values)),
                    'best': model_names[np.argmin(values) if 'perplexity' in metric else np.argmax(values)],
                    'worst': model_names[np.argmax(values) if 'perplexity' in metric else np.argmin(values)]
                }
        
        return comparison_results


def create_test_data(tokenizer: JadioTokenizer, 
                    num_samples: int = 100) -> Dict[str, Any]:
    """
    Create test data for evaluation.
    
    Args:
        tokenizer: Tokenizer for processing
        num_samples: Number of test samples
        
    Returns:
        Test data dictionary
    """
    # Sample prompts for generation
    prompts = [
        "The future of artificial intelligence is",
        "In a world where technology dominates,",
        "The most important lesson I learned was",
        "Climate change represents a challenge that",
        "The key to happiness lies in",
        "Scientific breakthroughs have shown us that",
        "The evolution of human society depends on",
        "In the digital age, privacy means",
        "The greatest innovations come from",
        "Environmental sustainability requires"
    ] * (num_samples // 10 + 1)
    
    prompts = prompts[:num_samples]
    
    # Sample references (for BLEU/ROUGE calculation)
    references = [
        "The future of artificial intelligence is bright and full of possibilities for humanity.",
        "In a world where technology dominates, human connections become more precious than ever.",
        "The most important lesson I learned was to never give up on my dreams.",
        "Climate change represents a challenge that requires global cooperation and immediate action.",
        "The key to happiness lies in appreciating what we have and connecting with others.",
        "Scientific breakthroughs have shown us that the universe is far more complex than we imagined.",
        "The evolution of human society depends on our ability to adapt and learn from history.",
        "In the digital age, privacy means having control over our personal information.",
        "The greatest innovations come from combining creativity with scientific rigor.",
        "Environmental sustainability requires balancing economic growth with ecological preservation."
    ] * (num_samples // 10 + 1)
    
    references = references[:num_samples]
    
    return {
        'prompts': prompts,
        'references': references
    }


def main():
    """Test the evaluation module."""
    print("[Jadio] Testing Evaluation Module...")
    
    # This test requires a trained model, so we'll create a minimal test
    try:
        from ..config.jadio_config import get_jadio_small_config
        from ..tokenizer.jadio_tokenizer import JadioTokenizer
        
        # Create small model for testing
        config = get_jadio_small_config()
        model = JadioLMHeadModel(config)
        
        # Try to load tokenizer
        try:
            tokenizer = JadioTokenizer.from_pretrained("modelling/jadio01")
        except:
            # Create dummy tokenizer for testing
            tokenizer = type('DummyTokenizer', (), {
                'encode': lambda self, text, add_special_tokens=True: list(range(min(len(text.split()), 20))),
                'decode': lambda self, tokens, skip_special_tokens=True: ' '.join([f"word_{t}" for t in tokens]),
                '__len__': lambda self: 1000,
                'eos_token_id': 0,
                'pad_token_id': 0,
            })()
        
        print(f"✓ Created test model: {model.get_num_params() / 1e6:.1f}M parameters")
        
        # Create evaluator
        evaluator = JadioEvaluator(model, tokenizer, device='cpu')
        print("✓ Created evaluator")
        
        # Create test data
        test_data = create_test_data(tokenizer, num_samples=5)
        print(f"✓ Created test data: {len(test_data['prompts'])} prompts")
        
        # Test generation evaluation
        print("\n--- Testing Generation Evaluation ---")
        generation_results = evaluator.evaluate_generation(
            test_data['prompts'][:3],
            generation_config={'max_new_tokens': 20, 'do_sample': False},
            reference_texts=test_data['references'][:3]
        )
        
        print("Generation metrics:")
        for key, value in generation_results['metrics'].items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}")
        
        print(f"✓ Generated {len(generation_results['samples'])} samples")
        
        # Test benchmark
        print("\n--- Testing Benchmark ---")
        benchmark_results = evaluator.run_benchmark(
            "test_benchmark",
            test_data,
            output_dir="test_evaluation_results"
        )
        
        print(f"✓ Benchmark completed: {len(benchmark_results.metrics)} metrics")
        print("Key metrics:")
        for key, value in list(benchmark_results.metrics.items())[:5]:
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}")
        
        # Clean up
        import shutil
        if Path("test_evaluation_results").exists():
            shutil.rmtree("test_evaluation_results")
        
        print("\n[Jadio] Evaluation module test completed!")
        
    except Exception as e:
        print(f"❌ Evaluation test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
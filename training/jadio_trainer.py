"""
Trainer class for the Jadio LLM.
Separated from the main training script for modularity.
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import math
from tqdm import tqdm
from pathlib import Path
from typing import Optional, Dict, Any

from ..scripts.jadio_utilities import setup_logger, save_checkpoint, load_checkpoint
from ..config.jadio_config import JadioConfig
from ..modelling.jadio01.jadio_decoder_transformer import JadioLMHeadModel

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class JadioTrainer:
    """
    Main trainer class for Jadio LLM.
    
    Handles model training, validation, checkpointing, and logging.
    Designed to be modular and reusable across different training scripts.
    """
    
    def __init__(self, 
                 config: JadioConfig, 
                 model_name: str = "jadio_model",
                 checkpoint_dir: Optional[str] = None,
                 logs_dir: Optional[str] = None):
        """
        Initialize the trainer.
        
        Args:
            config: Model configuration
            model_name: Name for the model (used in logging and checkpoints)
            checkpoint_dir: Directory to save checkpoints (default: checkpoints/{model_name})
            logs_dir: Directory to save logs (default: logs/{model_name})
        """
        self.config = config
        self.model_name = model_name
        self.device = torch.device(config.device)
        
        # Setup logging
        self.logger = setup_logger(f"jadio_trainer_{model_name}")
        self.logger.info(f"Initializing Jadio Trainer for {model_name}")
        self.logger.info(f"Device: {self.device}")
        
        # Initialize model
        self.model = JadioLMHeadModel(config).to(self.device)
        self.logger.info(f"Model parameters: {self.model.get_num_params() / 1e6:.1f}M")
        
        # Training state
        self.step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        self.training_history = []
        
        # Create directories
        self.checkpoint_dir = Path(checkpoint_dir or f"checkpoints/{model_name}")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.logs_dir = Path(logs_dir or f"logs/{model_name}")
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Optimizer and scheduler (will be created during training)
        self.optimizer = None
        self.scheduler = None
    
    def create_optimizer(self, 
                        learning_rate: float = 3e-4, 
                        weight_decay: float = 0.1, 
                        betas: tuple = (0.9, 0.95)) -> torch.optim.Optimizer:
        """
        Create AdamW optimizer with weight decay applied only to 2D parameters.
        
        This follows the GPT-2/GPT-3 approach where only 2D parameters (weight matrices)
        get weight decay, while 1D parameters (biases, layer norms) don't.
        
        Args:
            learning_rate: Learning rate
            weight_decay: Weight decay coefficient
            betas: Adam beta parameters
            
        Returns:
            Configured AdamW optimizer
        """
        # Separate parameters into decay and no_decay groups
        decay = set()
        no_decay = set()
        
        # Parameters that should have weight decay
        whitelist_weight_modules = (torch.nn.Linear,)
        # Parameters that should NOT have weight decay
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        
        for mn, m in self.model.named_modules():
            for pn, p in m.named_parameters():
                fpn = f'{mn}.{pn}' if mn else pn  # full param name
                
                if pn.endswith('bias'):
                    # All biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # Weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # Weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
        
        # Validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.model.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        
        assert len(inter_params) == 0, f"Parameters {inter_params} in both decay/no_decay sets!"
        assert len(param_dict.keys() - union_params) == 0, f"Parameters {param_dict.keys() - union_params} not categorized!"
        
        # Create optimizer groups
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        
        # Use fused AdamW if available (faster on modern GPUs)
        use_fused = (self.device.type == 'cuda') and ('fused' in torch.optim.AdamW.__doc__)
        self.logger.info(f"Using fused AdamW: {use_fused}")
        
        extra_args = {'fused': True} if use_fused else {}
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        
        self.optimizer = optimizer
        return optimizer
    
    def create_scheduler(self, 
                        optimizer: torch.optim.Optimizer,
                        warmup_steps: int = 1000, 
                        max_steps: int = 10000,
                        min_lr_ratio: float = 0.1) -> torch.optim.lr_scheduler.LambdaLR:
        """
        Create learning rate scheduler with linear warmup and cosine decay.
        
        Args:
            optimizer: Optimizer to schedule
            warmup_steps: Number of warmup steps
            max_steps: Total number of training steps
            min_lr_ratio: Minimum learning rate as ratio of initial LR
            
        Returns:
            Learning rate scheduler
        """
        def lr_lambda(step):
            if step < warmup_steps:
                # Linear warmup
                return step / warmup_steps
            else:
                # Cosine decay
                progress = (step - warmup_steps) / (max_steps - warmup_steps)
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                return min_lr_ratio + (1 - min_lr_ratio) * cosine_decay
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        self.scheduler = scheduler
        return scheduler
    
    def estimate_loss(self, 
                     dataloader: DataLoader, 
                     eval_steps: int = 100) -> float:
        """
        Estimate loss on validation set.
        
        Args:
            dataloader: Validation dataloader
            eval_steps: Maximum number of steps to evaluate
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        losses = []
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= eval_steps:
                    break
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                
                # Forward pass
                loss, _ = self.model(input_ids, labels=input_ids)
                losses.append(loss.item())
        
        self.model.train()
        avg_loss = sum(losses) / len(losses) if losses else float('inf')
        return avg_loss
    
    def train_step(self, 
                   batch: Dict[str, torch.Tensor], 
                   grad_clip: float = 1.0) -> float:
        """
        Single training step.
        
        Args:
            batch: Batch of training data
            grad_clip: Gradient clipping threshold
            
        Returns:
            Training loss for this step
        """
        # Move data to device
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch.get('attention_mask', None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        # Forward pass
        loss, logits = self.model(input_ids, labels=input_ids)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
        
        # Optimizer step
        if self.optimizer is not None:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
        
        # Scheduler step
        if self.scheduler is not None:
            self.scheduler.step()
        
        self.step += 1
        return loss.item()
    
    def save_checkpoint(self, 
                       filepath: str, 
                       is_best: bool = False,
                       extra_data: Optional[Dict[str, Any]] = None) -> None:
        """
        Save model checkpoint.
        
        Args:
            filepath: Path to save checkpoint
            is_best: Whether this is the best model so far
            extra_data: Additional data to save in checkpoint
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config.to_dict(),
            'step': self.step,
            'epoch': self.epoch,
            'best_loss': self.best_loss,
            'training_history': self.training_history,
        }
        
        if self.optimizer is not None:
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        if extra_data:
            checkpoint.update(extra_data)
        
        torch.save(checkpoint, filepath)
        self.logger.info(f"Saved checkpoint to {filepath}")
        
        if is_best:
            best_path = str(filepath).replace('.pt', '_best.pt')
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best checkpoint to {best_path}")
    
    def load_checkpoint(self, 
                       filepath: str, 
                       load_optimizer: bool = True,
                       load_scheduler: bool = True) -> Dict[str, Any]:
        """
        Load model checkpoint.
        
        Args:
            filepath: Path to checkpoint file
            load_optimizer: Whether to load optimizer state
            load_scheduler: Whether to load scheduler state
            
        Returns:
            Checkpoint data dictionary
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.step = checkpoint.get('step', 0)
        self.epoch = checkpoint.get('epoch', 0)
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        self.training_history = checkpoint.get('training_history', [])
        
        if load_optimizer and self.optimizer is not None and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if load_scheduler and self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.logger.info(f"Loaded checkpoint from {filepath}")
        self.logger.info(f"Resuming from step {self.step}, epoch {self.epoch}")
        
        return checkpoint
    
    def train(self, 
              train_dataloader: DataLoader,
              val_dataloader: Optional[DataLoader] = None,
              num_epochs: int = 3,
              learning_rate: float = 3e-4,
              weight_decay: float = 0.1,
              warmup_steps: int = 1000,
              eval_interval: int = 500,
              eval_steps: int = 100,
              save_interval: int = 1000,
              grad_clip: float = 1.0,
              resume_from: Optional[str] = None,
              use_wandb: bool = False,
              wandb_project: str = "jadio-llm",
              wandb_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main training loop.
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader (optional)
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            weight_decay: Weight decay for AdamW
            warmup_steps: Number of warmup steps for scheduler
            eval_interval: Steps between evaluations
            eval_steps: Number of steps for evaluation
            save_interval: Steps between saving checkpoints
            grad_clip: Gradient clipping threshold
            resume_from: Path to checkpoint to resume from
            use_wandb: Whether to use Weights & Biases logging
            wandb_project: W&B project name
            wandb_config: Additional W&B config
            
        Returns:
            Training history and final metrics
        """
        # Initialize wandb
        if use_wandb and WANDB_AVAILABLE:
            wandb_config = wandb_config or {}
            wandb_config.update({
                "model_name": self.model_name,
                "model_params": self.model.get_num_params(),
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "warmup_steps": warmup_steps,
                "num_epochs": num_epochs,
                "batch_size": train_dataloader.batch_size,
                **self.config.to_dict()
            })
            
            wandb.init(
                project=wandb_project,
                name=f"{self.model_name}_run_{int(time.time())}",
                config=wandb_config
            )
        
        # Create optimizer and scheduler
        max_steps = len(train_dataloader) * num_epochs
        self.create_optimizer(learning_rate, weight_decay)
        self.create_scheduler(self.optimizer, warmup_steps, max_steps)
        
        # Resume from checkpoint if specified
        if resume_from and os.path.exists(resume_from):
            self.load_checkpoint(resume_from)
        
        self.logger.info("Starting training...")
        self.logger.info(f"Total epochs: {num_epochs}")
        self.logger.info(f"Steps per epoch: {len(train_dataloader)}")
        self.logger.info(f"Total steps: {max_steps}")
        
        self.model.train()
        start_time = time.time()
        
        # Training loop
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            epoch_losses = []
            
            pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch_idx, batch in enumerate(pbar):
                # Training step
                loss = self.train_step(batch, grad_clip)
                epoch_losses.append(loss)
                
                # Update progress bar
                current_lr = self.scheduler.get_last_lr()[0] if self.scheduler else learning_rate
                pbar.set_postfix({
                    'loss': f'{loss:.4f}',
                    'lr': f'{current_lr:.2e}',
                    'step': self.step
                })
                
                # Log to wandb
                if use_wandb and WANDB_AVAILABLE:
                    wandb.log({
                        'train_loss': loss,
                        'learning_rate': current_lr,
                        'step': self.step,
                        'epoch': epoch
                    })
                
                # Evaluation
                if val_dataloader and self.step % eval_interval == 0:
                    val_loss = self.estimate_loss(val_dataloader, eval_steps)
                    self.logger.info(f"Step {self.step}: val_loss = {val_loss:.4f}")
                    
                    if use_wandb and WANDB_AVAILABLE:
                        wandb.log({'val_loss': val_loss, 'step': self.step})
                    
                    # Save best model
                    if val_loss < self.best_loss:
                        self.best_loss = val_loss
                        best_path = self.checkpoint_dir / f"step_{self.step}_best.pt"
                        self.save_checkpoint(str(best_path), is_best=True)
                
                # Save checkpoint
                if self.step % save_interval == 0:
                    checkpoint_path = self.checkpoint_dir / f"step_{self.step}.pt"
                    self.save_checkpoint(str(checkpoint_path))
            
            # End of epoch logging
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            elapsed = time.time() - start_time
            
            epoch_data = {
                'epoch': epoch,
                'avg_loss': avg_epoch_loss,
                'elapsed_time': elapsed,
                'step': self.step
            }
            self.training_history.append(epoch_data)
            
            self.logger.info(f"Epoch {epoch+1} completed: avg_loss = {avg_epoch_loss:.4f}, time = {elapsed:.1f}s")
            
            if use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    'epoch_loss': avg_epoch_loss,
                    'epoch': epoch,
                    'epoch_time': elapsed
                })
        
        # Final save
        final_path = self.checkpoint_dir / "final_model.pt"
        self.save_checkpoint(str(final_path))
        
        total_time = time.time() - start_time
        self.logger.info("Training completed!")
        self.logger.info(f"Total training time: {total_time:.1f}s")
        
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({'total_training_time': total_time})
            wandb.finish()
        
        return {
            'training_history': self.training_history,
            'final_loss': avg_epoch_loss,
            'total_time': total_time,
            'total_steps': self.step,
            'best_loss': self.best_loss
        }


def main():
    """Test the trainer."""
    print("[Jadio] Testing Trainer...")
    
    from ..config.jadio_config import get_jadio_small_config
    from ..data.jadio_dataset_loader import create_dummy_texts, TextDataset, create_dataloader
    from ..tokenizer.jadio_tokenizer import JadioTokenizer
    
    # Create small config for testing
    config = get_jadio_small_config()
    
    # Create trainer
    trainer = JadioTrainer(config, "test_trainer")
    print(f"✓ Created trainer with {trainer.model.get_num_params() / 1e6:.1f}M param model")
    
    # Create dummy dataset
    texts = create_dummy_texts(50, 20, 100)
    
    # For testing, create a simple tokenizer or use dummy data
    try:
        tokenizer = JadioTokenizer.from_pretrained("modelling/jadio01")
    except:
        # Create dummy tokenizer for testing
        tokenizer = type('DummyTokenizer', (), {
            'encode': lambda self, text, add_special_tokens=True: list(range(min(len(text), 50))),
            'eos_token_id': 0,
            'pad_token_id': 0,
        })()
    
    dataset = TextDataset(texts, tokenizer, max_length=64)
    dataloader = create_dataloader(dataset, batch_size=4)
    
    print(f"✓ Created dataset with {len(dataset)} samples")
    
    # Test training for a few steps
    history = trainer.train(
        train_dataloader=dataloader,
        num_epochs=1,
        learning_rate=1e-4,
        eval_interval=10,
        save_interval=20
    )
    
    print(f"✓ Training completed. Final loss: {history['final_loss']:.4f}")
    print("[Jadio] Trainer test completed!")


if __name__ == "__main__":
    main()
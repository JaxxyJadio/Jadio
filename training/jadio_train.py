#!/usr/bin/env python3
"""
Training script for the Jadio LLM.
"""
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import time
import math
from tqdm import tqdm
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.jadio_config import JadioConfig, get_jadio_50m_config, get_jadio_small_config
from modelling.jadio01.jadio_decoder_transformer import JadioLMHeadModel
from scripts.jadio_utilities import setup_logger, set_seed, save_checkpoint, load_checkpoint

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class JadioTrainer:
    """Main trainer class for Jadio LLM."""
    
    def __init__(self, config: JadioConfig, model_name: str = "jadio_50m"):
        """
        Initialize the trainer.
        
        Args:
            config: Model configuration
            model_name: Name for the model (used in logging and checkpoints)
        """
        self.config = config
        self.model_name = model_name
        self.device = torch.device(config.device)
        
        # Setup logging
        self.logger = setup_logger("jadio_trainer")
        self.logger.info(f"Initializing Jadio Trainer for {model_name}")
        self.logger.info(f"Device: {self.device}")
        
        # Initialize model
        self.model = JadioLMHeadModel(config).to(self.device)
        self.logger.info(f"Model parameters: {self.model.get_num_params() / 1e6:.1f}M")
        
        # Training state
        self.step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
        # Create directories
        self.checkpoint_dir = Path("checkpoints") / model_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.logs_dir = Path("logs") / model_name
        self.logs_dir.mkdir(parents=True, exist_ok=True)
    
    def create_optimizer(self, learning_rate: float = 3e-4, weight_decay: float = 0.1, 
                        betas: tuple = (0.9, 0.95), device_type: str = 'cuda'):
        """
        Create AdamW optimizer with weight decay applied only to 2D parameters.
        """
        # Separate parameters into decay and no_decay groups
        decay = set()
        no_decay = set()
        
        # Parameters that should not be decayed
        whitelist_weight_modules = (torch.nn.Linear,)
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
        assert len(inter_params) == 0, f"Parameters {inter_params} made it into both decay/no_decay sets!"
        assert len(param_dict.keys() - union_params) == 0, f"Parameters {param_dict.keys() - union_params} were not separated into either decay/no_decay set!"
        
        # Create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        
        # Use fused AdamW if available (faster)
        use_fused = (device_type == 'cuda') and ('fused' in torch.optim.AdamW.__doc__)
        self.logger.info(f"Using fused AdamW: {use_fused}")
        
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, fused=use_fused)
        return optimizer
    
    def create_scheduler(self, optimizer, warmup_steps: int = 1000, max_steps: int = 10000):
        """Create learning rate scheduler with warmup."""
        def lr_lambda(step):
            if step < warmup_steps:
                # Linear warmup
                return step / warmup_steps
            else:
                # Cosine decay
                progress = (step - warmup_steps) / (max_steps - warmup_steps)
                return 0.5 * (1 + math.cos(math.pi * progress))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return scheduler
    
    def estimate_loss(self, dataloader, eval_steps: int = 100):
        """Estimate loss on validation set."""
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
        return sum(losses) / len(losses) if losses else float('inf')
    
    def train_step(self, batch, optimizer, scheduler=None, grad_clip: float = 1.0):
        """Single training step."""
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
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
        # Scheduler step
        if scheduler is not None:
            scheduler.step()
        
        self.step += 1
        return loss.item()
    
    def save_model(self, path: str, optimizer=None, scheduler=None, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config.to_dict(),
            'step': self.step,
            'epoch': self.epoch,
            'best_loss': self.best_loss,
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        torch.save(checkpoint, path)
        self.logger.info(f"Saved checkpoint to {path}")
        
        if is_best:
            best_path = str(path).replace('.pt', '_best.pt')
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best checkpoint to {best_path}")
    
    def load_model(self, path: str, optimizer=None, scheduler=None):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.step = checkpoint.get('step', 0)
        self.epoch = checkpoint.get('epoch', 0)
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.logger.info(f"Loaded checkpoint from {path}")
        self.logger.info(f"Resuming from step {self.step}, epoch {self.epoch}")
    
    def train(self, 
              train_dataloader: DataLoader,
              val_dataloader: DataLoader = None,
              num_epochs: int = 3,
              learning_rate: float = 3e-4,
              weight_decay: float = 0.1,
              warmup_steps: int = 1000,
              eval_interval: int = 500,
              eval_steps: int = 100,
              save_interval: int = 1000,
              grad_clip: float = 1.0,
              resume_from: str = None,
              use_wandb: bool = False,
              wandb_project: str = "jadio-llm"):
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
        """
        # Initialize wandb
        if use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=wandb_project,
                name=f"{self.model_name}_run_{int(time.time())}",
                config={
                    "model_name": self.model_name,
                    "model_params": self.model.get_num_params(),
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "warmup_steps": warmup_steps,
                    "num_epochs": num_epochs,
                    "batch_size": train_dataloader.batch_size,
                    **self.config.to_dict()
                }
            )
        
        # Create optimizer and scheduler
        max_steps = len(train_dataloader) * num_epochs
        optimizer = self.create_optimizer(learning_rate, weight_decay)
        scheduler = self.create_scheduler(optimizer, warmup_steps, max_steps)
        
        # Resume from checkpoint if specified
        if resume_from and os.path.exists(resume_from):
            self.load_model(resume_from, optimizer, scheduler)
        
        self.logger.info("Starting training...")
        self.logger.info(f"Total epochs: {num_epochs}")
        self.logger.info(f"Steps per epoch: {len(train_dataloader)}")
        self.logger.info(f"Total steps: {max_steps}")
        
        self.model.train()
        start_time = time.time()
        
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            epoch_losses = []
            
            # Training loop
            pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch_idx, batch in enumerate(pbar):
                # Training step
                loss = self.train_step(batch, optimizer, scheduler, grad_clip)
                epoch_losses.append(loss)
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss:.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}',
                    'step': self.step
                })
                
                # Log to wandb
                if use_wandb and WANDB_AVAILABLE:
                    wandb.log({
                        'train_loss': loss,
                        'learning_rate': scheduler.get_last_lr()[0],
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
                        self.save_model(best_path, optimizer, scheduler, is_best=True)
                
                # Save checkpoint
                if self.step % save_interval == 0:
                    checkpoint_path = self.checkpoint_dir / f"step_{self.step}.pt"
                    self.save_model(checkpoint_path, optimizer, scheduler)
            
            # End of epoch logging
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            elapsed = time.time() - start_time
            self.logger.info(f"Epoch {epoch+1} completed: avg_loss = {avg_epoch_loss:.4f}, time = {elapsed:.1f}s")
            
            if use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    'epoch_loss': avg_epoch_loss,
                    'epoch': epoch,
                    'epoch_time': elapsed
                })
        
        # Final save
        final_path = self.checkpoint_dir / "final_model.pt"
        self.save_model(final_path, optimizer, scheduler)
        
        self.logger.info("Training completed!")
        total_time = time.time() - start_time
        self.logger.info(f"Total training time: {total_time:.1f}s")
        
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({'total_training_time': total_time})
            wandb.finish()


def create_dummy_dataset(vocab_size: int = 50257, seq_len: int = 1024, num_samples: int = 1000):
    """Create a dummy dataset for testing."""
    from torch.utils.data import TensorDataset
    
    # Generate random token sequences
    input_ids = torch.randint(0, vocab_size, (num_samples, seq_len))
    
    # Create attention masks (all ones for simplicity)
    attention_mask = torch.ones_like(input_ids)
    
    dataset = TensorDataset(input_ids, attention_mask)
    return dataset


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Jadio LLM")
    parser.add_argument("--model_size", choices=["small", "50m"], default="small",
                       help="Model size to train")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.1,
                       help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=100,
                       help="Number of warmup steps")
    parser.add_argument("--eval_interval", type=int, default=100,
                       help="Steps between evaluations")
    parser.add_argument("--save_interval", type=int, default=500,
                       help="Steps between saving checkpoints")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                       help="Gradient clipping threshold")
    parser.add_argument("--resume_from", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--use_wandb", action="store_true",
                       help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="jadio-llm",
                       help="W&B project name")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--dummy_data", action="store_true",
                       help="Use dummy data for testing")
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Get configuration
    if args.model_size == "small":
        config = get_jadio_small_config()
        model_name = "jadio_small"
    else:
        config = get_jadio_50m_config()
        model_name = "jadio_50m"
    
    print(f"Training {model_name} with {config.get_model_params() / 1e6:.1f}M parameters")
    
    # Create trainer
    trainer = JadioTrainer(config, model_name)
    
    # Create datasets
    if args.dummy_data:
        print("Using dummy data for testing...")
        train_dataset = create_dummy_dataset(config.vocab_size, config.n_ctx, 1000)
        val_dataset = create_dummy_dataset(config.vocab_size, config.n_ctx, 200)
    else:
        # TODO: Implement real data loading
        print("Real data loading not implemented yet. Using dummy data.")
        train_dataset = create_dummy_dataset(config.vocab_size, config.n_ctx, 1000)
        val_dataset = create_dummy_dataset(config.vocab_size, config.n_ctx, 200)
    
    # Create data loaders
    def collate_fn(batch):
        input_ids = torch.stack([item[0] for item in batch])
        attention_mask = torch.stack([item[1] for item in batch])
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Start training
    trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        grad_clip=args.grad_clip,
        resume_from=args.resume_from,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project
    )


if __name__ == "__main__":
    main()
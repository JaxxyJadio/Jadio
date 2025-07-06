"""
Learning rate schedulers for the Jadio LLM.
"""
import torch
import torch.optim as optim
import math
from typing import Optional, Union, List, Callable
import warnings


class LinearWarmupCosineDecayLR(optim.lr_scheduler.LambdaLR):
    """
    Learning rate scheduler with linear warmup followed by cosine decay.
    
    This is the standard scheduler used in most modern LLM training.
    """
    
    def __init__(self,
                 optimizer: optim.Optimizer,
                 warmup_steps: int,
                 max_steps: int,
                 min_lr_ratio: float = 0.1,
                 last_epoch: int = -1):
        """
        Initialize the scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            warmup_steps: Number of warmup steps
            max_steps: Total number of training steps
            min_lr_ratio: Minimum learning rate as ratio of initial LR
            last_epoch: Last epoch for resuming training
        """
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr_ratio = min_lr_ratio
        
        def lr_lambda(step):
            if step < self.warmup_steps:
                # Linear warmup
                return step / self.warmup_steps
            else:
                # Cosine decay
                progress = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                return self.min_lr_ratio + (1 - self.min_lr_ratio) * cosine_decay
        
        super().__init__(optimizer, lr_lambda, last_epoch)


class LinearWarmupLinearDecayLR(optim.lr_scheduler.LambdaLR):
    """
    Learning rate scheduler with linear warmup followed by linear decay.
    """
    
    def __init__(self,
                 optimizer: optim.Optimizer,
                 warmup_steps: int,
                 max_steps: int,
                 min_lr_ratio: float = 0.0,
                 last_epoch: int = -1):
        """
        Initialize the scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            warmup_steps: Number of warmup steps
            max_steps: Total number of training steps
            min_lr_ratio: Minimum learning rate as ratio of initial LR
            last_epoch: Last epoch for resuming training
        """
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr_ratio = min_lr_ratio
        
        def lr_lambda(step):
            if step < self.warmup_steps:
                # Linear warmup
                return step / self.warmup_steps
            else:
                # Linear decay
                progress = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
                return self.min_lr_ratio + (1 - self.min_lr_ratio) * (1 - progress)
        
        super().__init__(optimizer, lr_lambda, last_epoch)


class InverseSqrtLR(optim.lr_scheduler.LambdaLR):
    """
    Inverse square root learning rate scheduler.
    
    Used in some transformer implementations, particularly Attention is All You Need.
    """
    
    def __init__(self,
                 optimizer: optim.Optimizer,
                 warmup_steps: int,
                 last_epoch: int = -1):
        """
        Initialize the scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            warmup_steps: Number of warmup steps
            last_epoch: Last epoch for resuming training
        """
        self.warmup_steps = warmup_steps
        
        def lr_lambda(step):
            if step < self.warmup_steps:
                # Linear warmup
                return step / self.warmup_steps
            else:
                # Inverse square root decay
                return math.sqrt(self.warmup_steps / step)
        
        super().__init__(optimizer, lr_lambda, last_epoch)


class PolynomialDecayLR(optim.lr_scheduler.LambdaLR):
    """
    Polynomial decay learning rate scheduler.
    """
    
    def __init__(self,
                 optimizer: optim.Optimizer,
                 warmup_steps: int,
                 max_steps: int,
                 power: float = 1.0,
                 min_lr_ratio: float = 0.0,
                 last_epoch: int = -1):
        """
        Initialize the scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            warmup_steps: Number of warmup steps
            max_steps: Total number of training steps
            power: Power for polynomial decay
            min_lr_ratio: Minimum learning rate as ratio of initial LR
            last_epoch: Last epoch for resuming training
        """
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.power = power
        self.min_lr_ratio = min_lr_ratio
        
        def lr_lambda(step):
            if step < self.warmup_steps:
                # Linear warmup
                return step / self.warmup_steps
            else:
                # Polynomial decay
                progress = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
                decay_factor = (1 - progress) ** self.power
                return self.min_lr_ratio + (1 - self.min_lr_ratio) * decay_factor
        
        super().__init__(optimizer, lr_lambda, last_epoch)


class ExponentialDecayLR(optim.lr_scheduler.LambdaLR):
    """
    Exponential decay learning rate scheduler with warmup.
    """
    
    def __init__(self,
                 optimizer: optim.Optimizer,
                 warmup_steps: int,
                 decay_rate: float = 0.95,
                 decay_steps: int = 1000,
                 last_epoch: int = -1):
        """
        Initialize the scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            warmup_steps: Number of warmup steps
            decay_rate: Decay rate for exponential decay
            decay_steps: Steps between decay applications
            last_epoch: Last epoch for resuming training
        """
        self.warmup_steps = warmup_steps
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        
        def lr_lambda(step):
            if step < self.warmup_steps:
                # Linear warmup
                return step / self.warmup_steps
            else:
                # Exponential decay
                decay_epochs = (step - self.warmup_steps) // self.decay_steps
                return self.decay_rate ** decay_epochs
        
        super().__init__(optimizer, lr_lambda, last_epoch)


class CyclicalLR(optim.lr_scheduler.LambdaLR):
    """
    Cyclical learning rate scheduler.
    
    Cycles between a minimum and maximum learning rate.
    """
    
    def __init__(self,
                 optimizer: optim.Optimizer,
                 cycle_length: int,
                 min_lr_ratio: float = 0.1,
                 last_epoch: int = -1):
        """
        Initialize the scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            cycle_length: Length of each cycle in steps
            min_lr_ratio: Minimum learning rate as ratio of max LR
            last_epoch: Last epoch for resuming training
        """
        self.cycle_length = cycle_length
        self.min_lr_ratio = min_lr_ratio
        
        def lr_lambda(step):
            cycle_position = (step % self.cycle_length) / self.cycle_length
            # Cosine function for smooth cycling
            cosine_factor = 0.5 * (1 + math.cos(math.pi * cycle_position))
            return self.min_lr_ratio + (1 - self.min_lr_ratio) * cosine_factor
        
        super().__init__(optimizer, lr_lambda, last_epoch)


class OneCycleLR(optim.lr_scheduler.LambdaLR):
    """
    One cycle learning rate scheduler.
    
    Single cycle that goes up to a maximum then back down, following the
    "Super-Convergence" paper approach.
    """
    
    def __init__(self,
                 optimizer: optim.Optimizer,
                 max_lr_ratio: float = 10.0,
                 total_steps: int = 1000,
                 pct_start: float = 0.3,
                 div_factor: float = 25.0,
                 final_div_factor: float = 1e4,
                 last_epoch: int = -1):
        """
        Initialize the scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            max_lr_ratio: Maximum LR as ratio of initial LR
            total_steps: Total number of training steps
            pct_start: Percentage of cycle spent increasing LR
            div_factor: Initial LR divisor
            final_div_factor: Final LR divisor
            last_epoch: Last epoch for resuming training
        """
        self.max_lr_ratio = max_lr_ratio
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        
        def lr_lambda(step):
            if step < self.pct_start * self.total_steps:
                # Increasing phase
                progress = step / (self.pct_start * self.total_steps)
                start_lr = 1.0 / self.div_factor
                return start_lr + (self.max_lr_ratio - start_lr) * progress
            else:
                # Decreasing phase
                progress = (step - self.pct_start * self.total_steps) / ((1 - self.pct_start) * self.total_steps)
                end_lr = 1.0 / self.final_div_factor
                return self.max_lr_ratio - (self.max_lr_ratio - end_lr) * progress
        
        super().__init__(optimizer, lr_lambda, last_epoch)


def get_scheduler(scheduler_name: str,
                 optimizer: optim.Optimizer,
                 **kwargs) -> optim.lr_scheduler.LRScheduler:
    """
    Factory function to create learning rate schedulers.
    
    Args:
        scheduler_name: Name of the scheduler
        optimizer: Optimizer to schedule
        **kwargs: Scheduler-specific arguments
        
    Returns:
        Configured learning rate scheduler
    """
    scheduler_name = scheduler_name.lower()
    
    if scheduler_name == 'cosine':
        return LinearWarmupCosineDecayLR(optimizer, **kwargs)
    elif scheduler_name == 'linear':
        return LinearWarmupLinearDecayLR(optimizer, **kwargs)
    elif scheduler_name == 'inverse_sqrt':
        return InverseSqrtLR(optimizer, **kwargs)
    elif scheduler_name == 'polynomial':
        return PolynomialDecayLR(optimizer, **kwargs)
    elif scheduler_name == 'exponential':
        return ExponentialDecayLR(optimizer, **kwargs)
    elif scheduler_name == 'cyclical':
        return CyclicalLR(optimizer, **kwargs)
    elif scheduler_name == 'onecycle':
        return OneCycleLR(optimizer, **kwargs)
    elif scheduler_name == 'constant':
        return optim.lr_scheduler.ConstantLR(optimizer, **kwargs)
    elif scheduler_name == 'step':
        return optim.lr_scheduler.StepLR(optimizer, **kwargs)
    elif scheduler_name == 'multistep':
        return optim.lr_scheduler.MultiStepLR(optimizer, **kwargs)
    elif scheduler_name == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


def plot_scheduler(scheduler: optim.lr_scheduler.LRScheduler,
                  steps: int,
                  initial_lr: float = 1e-3) -> None:
    """
    Plot learning rate schedule for visualization.
    
    Args:
        scheduler: Learning rate scheduler
        steps: Number of steps to plot
        initial_lr: Initial learning rate
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available for plotting")
        return
    
    lrs = []
    for step in range(steps):
        # Get learning rate at this step
        if hasattr(scheduler, 'lr_lambdas'):
            # For LambdaLR schedulers
            lr = initial_lr * scheduler.lr_lambdas[0](step)
        else:
            # For other schedulers, we need to actually step through
            # This is a simplified approach
            lr = initial_lr
        lrs.append(lr)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(steps), lrs)
    plt.xlabel('Training Steps')
    plt.ylabel('Learning Rate')
    plt.title(f'Learning Rate Schedule: {type(scheduler).__name__}')
    plt.grid(True, alpha=0.3)
    plt.show()


def get_lr_at_step(scheduler: optim.lr_scheduler.LRScheduler, 
                  step: int, 
                  initial_lr: float = 1e-3) -> float:
    """
    Get learning rate at a specific step without affecting scheduler state.
    
    Args:
        scheduler: Learning rate scheduler
        step: Step number
        initial_lr: Initial learning rate
        
    Returns:
        Learning rate at the specified step
    """
    if hasattr(scheduler, 'lr_lambdas'):
        return initial_lr * scheduler.lr_lambdas[0](step)
    else:
        # For non-lambda schedulers, this is trickier
        # We'd need to create a copy and step through it
        warnings.warn("Cannot easily get LR for this scheduler type")
        return initial_lr


def main():
    """Test the learning rate schedulers."""
    print("[Jadio] Testing Learning Rate Schedulers...")
    
    # Create a dummy optimizer
    import torch.nn as nn
    model = nn.Linear(10, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Test different schedulers
    schedulers_to_test = [
        ('cosine', {'warmup_steps': 100, 'max_steps': 1000}),
        ('linear', {'warmup_steps': 100, 'max_steps': 1000}),
        ('inverse_sqrt', {'warmup_steps': 100}),
        ('polynomial', {'warmup_steps': 100, 'max_steps': 1000, 'power': 2.0}),
        ('exponential', {'warmup_steps': 100, 'decay_rate': 0.95, 'decay_steps': 100}),
        ('onecycle', {'max_lr_ratio': 10.0, 'total_steps': 1000}),
    ]
    
    print(f"Testing {len(schedulers_to_test)} schedulers...")
    
    for sched_name, kwargs in schedulers_to_test:
        print(f"\n--- Testing {sched_name} scheduler ---")
        try:
            scheduler = get_scheduler(sched_name, optimizer, **kwargs)
            print(f"✓ Created {type(scheduler).__name__}")
            
            # Test a few steps
            initial_lr = optimizer.param_groups[0]['lr']
            print(f"Initial LR: {initial_lr:.6f}")
            
            # Show LR at different steps
            test_steps = [0, 50, 100, 200, 500, 1000]
            for step in test_steps:
                try:
                    lr = get_lr_at_step(scheduler, step, initial_lr)
                    print(f"  Step {step:4d}: LR = {lr:.6f}")
                except:
                    pass
            
        except Exception as e:
            print(f"❌ Failed to create {sched_name}: {e}")
    
    # Test actual scheduler stepping
    print("\n--- Testing scheduler stepping ---")
    scheduler = get_scheduler('cosine', optimizer, warmup_steps=10, max_steps=100)
    
    initial_lr = optimizer.param_groups[0]['lr']
    print(f"Starting LR: {initial_lr:.6f}")
    
    for step in range(20):
        current_lr = optimizer.param_groups[0]['lr']
        if step % 5 == 0:
            print(f"Step {step:2d}: LR = {current_lr:.6f}")
        scheduler.step()
    
    print("\n--- Testing scheduler state saving/loading ---")
    # Save scheduler state
    state_dict = scheduler.state_dict()
    print(f"✓ Saved scheduler state: {len(state_dict)} keys")
    
    # Create new scheduler and load state
    new_scheduler = get_scheduler('cosine', optimizer, warmup_steps=10, max_steps=100)
    new_scheduler.load_state_dict(state_dict)
    print(f"✓ Loaded scheduler state")
    
    # Verify they have the same state
    print(f"Original last_epoch: {scheduler.last_epoch}")
    print(f"Loaded last_epoch: {new_scheduler.last_epoch}")
    
    print("\n[Jadio] Learning rate scheduler test completed!")


if __name__ == "__main__":
    main()
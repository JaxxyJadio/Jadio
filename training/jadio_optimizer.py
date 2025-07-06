"""
Optimizer utilities for the Jadio LLM.
"""
import torch
import torch.nn as nn
from typing import Dict, List, Set, Tuple, Optional, Union
import math


def separate_weight_decay_params(model: nn.Module, 
                                weight_decay: float = 0.1) -> List[Dict[str, Union[List[torch.Tensor], float]]]:
    """
    Separate model parameters into those that should and shouldn't have weight decay.
    
    Following best practices from GPT-2/3 training:
    - 2D parameters (weight matrices) get weight decay
    - 1D parameters (biases, layer norms, embeddings) don't get weight decay
    
    Args:
        model: PyTorch model
        weight_decay: Weight decay coefficient
        
    Returns:
        List of parameter groups for optimizer
    """
    decay = set()
    no_decay = set()
    
    # Modules that should have weight decay on their weights
    whitelist_weight_modules = (torch.nn.Linear,)
    # Modules that should NOT have weight decay
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
    
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = f'{mn}.{pn}' if mn else pn  # full param name
            
            if pn.endswith('bias'):
                # All biases should not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # Weights of whitelist modules should be weight decayed
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                # Weights of blacklist modules should NOT be weight decayed
                no_decay.add(fpn)
    
    # Validate that we've categorized every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    
    assert len(inter_params) == 0, f"Parameters {inter_params} in both decay/no_decay sets!"
    assert len(param_dict.keys() - union_params) == 0, f"Parameters {param_dict.keys() - union_params} not categorized!"
    
    # Create parameter groups
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    
    return optim_groups


def create_adamw_optimizer(model: nn.Module,
                          learning_rate: float = 3e-4,
                          weight_decay: float = 0.1,
                          betas: Tuple[float, float] = (0.9, 0.95),
                          eps: float = 1e-8,
                          use_fused: Optional[bool] = None) -> torch.optim.AdamW:
    """
    Create AdamW optimizer with proper weight decay separation.
    
    Args:
        model: Model to optimize
        learning_rate: Learning rate
        weight_decay: Weight decay coefficient
        betas: Adam beta parameters (momentum terms)
        eps: Adam epsilon for numerical stability
        use_fused: Whether to use fused AdamW (auto-detect if None)
        
    Returns:
        Configured AdamW optimizer
    """
    param_groups = separate_weight_decay_params(model, weight_decay)
    
    # Auto-detect fused optimizer availability
    if use_fused is None:
        use_fused = torch.cuda.is_available() and 'fused' in torch.optim.AdamW.__doc__
    
    extra_args = {'fused': True} if use_fused else {}
    
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=learning_rate,
        betas=betas,
        eps=eps,
        **extra_args
    )
    
    return optimizer


def create_sgd_optimizer(model: nn.Module,
                        learning_rate: float = 0.01,
                        momentum: float = 0.9,
                        weight_decay: float = 0.1,
                        nesterov: bool = True) -> torch.optim.SGD:
    """
    Create SGD optimizer with momentum and weight decay.
    
    Args:
        model: Model to optimize
        learning_rate: Learning rate
        momentum: Momentum coefficient
        weight_decay: Weight decay coefficient
        nesterov: Whether to use Nesterov momentum
        
    Returns:
        Configured SGD optimizer
    """
    param_groups = separate_weight_decay_params(model, weight_decay)
    
    optimizer = torch.optim.SGD(
        param_groups,
        lr=learning_rate,
        momentum=momentum,
        nesterov=nesterov
    )
    
    return optimizer


class ScaledAdamW(torch.optim.Optimizer):
    """
    AdamW optimizer with parameter scaling for improved stability.
    
    This implements the scaled version of AdamW that's often used
    in large language model training for better numerical stability.
    """
    
    def __init__(self,
                 params,
                 lr: float = 1e-3,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8,
                 weight_decay: float = 1e-2,
                 scale_parameter: bool = True,
                 relative_step: bool = True):
        """
        Initialize ScaledAdamW optimizer.
        
        Args:
            params: Model parameters or parameter groups
            lr: Learning rate
            betas: Adam beta parameters
            eps: Epsilon for numerical stability
            weight_decay: Weight decay coefficient
            scale_parameter: Whether to scale parameters
            relative_step: Whether to use relative step size
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            scale_parameter=scale_parameter, relative_step=relative_step
        )
        super().__init__(params, defaults)
    
    def _get_lr(self, group, param_state):
        """Get learning rate for parameter group."""
        if group['relative_step']:
            min_step = 1e-6 * param_state['step'] if group['scale_parameter'] else 1e-2
            rel_step_sz = min(min_step, 1.0 / math.sqrt(param_state['step']))
            param_scale = 1.0
            if group['scale_parameter']:
                param_scale = max(group['eps'], param_state['RMS'])
            return param_scale * rel_step_sz
        else:
            return group['lr']
    
    def _get_options(self, group, param_shape):
        """Get optimizer options for parameter."""
        factored = len(param_shape) >= 2
        use_first_moment = group['betas'][0] > 0.0
        return factored, use_first_moment
    
    def _rms(self, tensor):
        """Root mean square."""
        return tensor.norm(2) / (tensor.numel() ** 0.5)
    
    def _approx_sq_grad(self, exp_avg_sq_row, exp_avg_sq_col):
        """Approximation of exponential moving average of square of gradient."""
        r_factor = ((exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True))
                   .rsqrt_().unsqueeze(-1).clamp_(0, math.inf))
        c_factor = (exp_avg_sq_col.rsqrt()).unsqueeze(0).clamp_(0, math.inf)
        return torch.mul(r_factor, c_factor)
    
    def step(self, closure=None):
        """Perform optimization step."""
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()
                
                state = self.state[p]
                grad_shape = grad.shape
                
                factored, use_first_moment = self._get_options(group, grad_shape)
                
                # State Initialization
                if len(state) == 0:
                    state['step'] = 0
                    
                    if use_first_moment:
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(grad).float()
                    if factored:
                        state['exp_avg_sq_row'] = torch.zeros(grad_shape[:-1]).float()
                        state['exp_avg_sq_col'] = torch.zeros(
                            grad_shape[:-2] + grad_shape[-1:]
                        ).float()
                    else:
                        state['exp_avg_sq'] = torch.zeros_like(grad).float()
                    
                    state['RMS'] = 0
                
                p_data_fp32 = p.data.float() if p.data.dtype in {torch.float16, torch.bfloat16} else p.data
                
                state['step'] += 1
                state['RMS'] = self._rms(p_data_fp32)
                
                lr = self._get_lr(group, state)
                
                beta2t = 1.0 - math.pow(group['betas'][1], state['step'])
                update = (grad**2) + group['eps']
                
                if factored:
                    exp_avg_sq_row = state['exp_avg_sq_row']
                    exp_avg_sq_col = state['exp_avg_sq_col']
                    
                    exp_avg_sq_row.mul_(group['betas'][1]).add_(
                        update.mean(dim=-1), alpha=1.0 - group['betas'][1]
                    )
                    exp_avg_sq_col.mul_(group['betas'][1]).add_(
                        update.mean(dim=-2), alpha=1.0 - group['betas'][1]
                    )
                    update = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
                    update.mul_(grad)
                else:
                    exp_avg_sq = state['exp_avg_sq']
                    
                    exp_avg_sq.mul_(group['betas'][1]).add_(update, alpha=1.0 - group['betas'][1])
                    update = exp_avg_sq.rsqrt().mul_(grad)
                
                update.div_((beta2t ** 0.5))
                
                if use_first_moment:
                    exp_avg = state['exp_avg']
                    exp_avg.mul_(group['betas'][0]).add_(update, alpha=1 - group['betas'][0])
                    update = exp_avg
                
                if group['weight_decay'] > 0:
                    p_data_fp32.mul_(1 - group['weight_decay'] * lr)
                
                p_data_fp32.add_(update, alpha=-lr)
                
                if p.data.dtype in {torch.float16, torch.bfloat16}:
                    p.data.copy_(p_data_fp32)
        
        return loss


def get_optimizer(optimizer_name: str, 
                 model: nn.Module, 
                 **kwargs) -> torch.optim.Optimizer:
    """
    Factory function to create optimizers.
    
    Args:
        optimizer_name: Name of optimizer ('adamw', 'sgd', 'scaled_adamw')
        model: Model to optimize
        **kwargs: Optimizer-specific arguments
        
    Returns:
        Configured optimizer
    """
    optimizer_name = optimizer_name.lower()
    
    if optimizer_name == 'adamw':
        return create_adamw_optimizer(model, **kwargs)
    elif optimizer_name == 'sgd':
        return create_sgd_optimizer(model, **kwargs)
    elif optimizer_name == 'scaled_adamw':
        param_groups = separate_weight_decay_params(model, kwargs.get('weight_decay', 0.1))
        return ScaledAdamW(param_groups, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def print_optimizer_info(optimizer: torch.optim.Optimizer) -> None:
    """
    Print information about the optimizer configuration.
    
    Args:
        optimizer: Optimizer to analyze
    """
    print(f"Optimizer: {type(optimizer).__name__}")
    print(f"Parameter groups: {len(optimizer.param_groups)}")
    
    total_params = 0
    for i, group in enumerate(optimizer.param_groups):
        group_params = sum(p.numel() for p in group['params'])
        total_params += group_params
        
        print(f"  Group {i}: {group_params:,} parameters")
        print(f"    Weight decay: {group.get('weight_decay', 'N/A')}")
        print(f"    Learning rate: {group.get('lr', 'N/A')}")
        
        # Show a few key parameters
        for key in ['betas', 'eps', 'momentum']:
            if key in group:
                print(f"    {key}: {group[key]}")
    
    print(f"Total parameters: {total_params:,}")


def main():
    """Test the optimizer utilities."""
    print("[Jadio] Testing Optimizer utilities...")
    
    # Create a simple test model
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(1000, 128)
            self.linear1 = nn.Linear(128, 256)
            self.ln = nn.LayerNorm(256)
            self.linear2 = nn.Linear(256, 1000)
        
        def forward(self, x):
            x = self.embedding(x)
            x = self.linear1(x)
            x = self.ln(x)
            x = self.linear2(x)
            return x
    
    model = TestModel()
    print(f"✓ Created test model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test parameter separation
    param_groups = separate_weight_decay_params(model, weight_decay=0.1)
    print(f"✓ Separated parameters into {len(param_groups)} groups")
    
    decay_params = sum(p.numel() for p in param_groups[0]['params'])
    no_decay_params = sum(p.numel() for p in param_groups[1]['params'])
    print(f"  - Weight decay: {decay_params:,} parameters")
    print(f"  - No weight decay: {no_decay_params:,} parameters")
    
    # Test optimizers
    print("\n--- Testing AdamW ---")
    adamw = create_adamw_optimizer(model, learning_rate=1e-3)
    print_optimizer_info(adamw)
    
    print("\n--- Testing SGD ---")
    sgd = create_sgd_optimizer(model, learning_rate=0.01)
    print_optimizer_info(sgd)
    
    print("\n--- Testing ScaledAdamW ---")
    scaled_adamw = get_optimizer('scaled_adamw', model, lr=1e-3)
    print_optimizer_info(scaled_adamw)
    
    # Test optimization step
    print("\n--- Testing optimization step ---")
    x = torch.randint(0, 1000, (4, 10))
    y = torch.randint(0, 1000, (4, 10, 1000))
    
    model.train()
    optimizer = adamw
    optimizer.zero_grad()
    
    output = model(x)
    loss = nn.CrossEntropyLoss()(output.view(-1, 1000), y.view(-1))
    loss.backward()
    
    print(f"Loss before step: {loss.item():.4f}")
    
    # Check gradients
    total_grad_norm = torch.norm(torch.stack([p.grad.norm() for p in model.parameters() if p.grad is not None]))
    print(f"Gradient norm: {total_grad_norm:.4f}")
    
    optimizer.step()
    print("✓ Optimization step completed")
    
    print("\n[Jadio] Optimizer utilities test completed!")


if __name__ == "__main__":
    main()
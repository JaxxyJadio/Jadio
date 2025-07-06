"""
Weights & Biases integration for the Jadio LLM project.
"""
import os
import time
from typing import Dict, Any, Optional, Union, List
import json
import tempfile
from pathlib import Path

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False


class JadioWandbLogger:
    """
    Weights & Biases logger for Jadio LLM training.
    
    Provides a clean interface for logging training metrics, model artifacts,
    and configuration to W&B.
    """
    
    def __init__(self,
                 project: str = "jadio-llm",
                 name: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None,
                 tags: Optional[List[str]] = None,
                 notes: Optional[str] = None,
                 group: Optional[str] = None,
                 job_type: Optional[str] = "train",
                 enabled: bool = True):
        """
        Initialize W&B logger.
        
        Args:
            project: W&B project name
            name: Run name (auto-generated if None)
            config: Configuration dictionary to log
            tags: List of tags for the run
            notes: Notes about the run
            group: Group name for organizing runs
            job_type: Type of job (train, eval, etc.)
            enabled: Whether to actually log to W&B
        """
        self.enabled = enabled and WANDB_AVAILABLE
        self.project = project
        self.run = None
        
        if not self.enabled:
            if not WANDB_AVAILABLE:
                print("⚠ wandb not available, logging disabled")
            else:
                print("⚠ W&B logging disabled")
            return
        
        # Generate name if not provided
        if name is None:
            timestamp = int(time.time())
            name = f"jadio_run_{timestamp}"
        
        # Initialize wandb run
        try:
            self.run = wandb.init(
                project=project,
                name=name,
                config=config or {},
                tags=tags,
                notes=notes,
                group=group,
                job_type=job_type,
                reinit=True
            )
            print(f"✓ W&B run initialized: {self.run.url}")
        except Exception as e:
            print(f"❌ Failed to initialize W&B: {e}")
            self.enabled = False
    
    def log(self, 
            metrics: Dict[str, Union[float, int]], 
            step: Optional[int] = None,
            commit: bool = True) -> None:
        """
        Log metrics to W&B.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Step number (auto-incremented if None)
            commit: Whether to commit the log entry
        """
        if not self.enabled or self.run is None:
            return
        
        try:
            wandb.log(metrics, step=step, commit=commit)
        except Exception as e:
            print(f"❌ Failed to log metrics: {e}")
    
    def log_model(self,
                  model_path: str,
                  name: Optional[str] = None,
                  aliases: Optional[List[str]] = None,
                  metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a model artifact to W&B.
        
        Args:
            model_path: Path to model checkpoint
            name: Artifact name
            aliases: List of aliases for the artifact
            metadata: Additional metadata
        """
        if not self.enabled or self.run is None:
            return
        
        try:
            artifact_name = name or f"model_{int(time.time())}"
            artifact = wandb.Artifact(
                name=artifact_name,
                type="model",
                description="Jadio LLM checkpoint",
                metadata=metadata or {}
            )
            
            artifact.add_file(model_path)
            
            self.run.log_artifact(artifact, aliases=aliases)
            print(f"✓ Logged model artifact: {artifact_name}")
            
        except Exception as e:
            print(f"❌ Failed to log model: {e}")
    
    def log_dataset(self,
                   dataset_path: str,
                   name: Optional[str] = None,
                   description: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a dataset artifact to W&B.
        
        Args:
            dataset_path: Path to dataset
            name: Artifact name
            description: Dataset description
            metadata: Additional metadata
        """
        if not self.enabled or self.run is None:
            return
        
        try:
            artifact_name = name or f"dataset_{int(time.time())}"
            artifact = wandb.Artifact(
                name=artifact_name,
                type="dataset",
                description=description or "Jadio LLM training dataset",
                metadata=metadata or {}
            )
            
            if os.path.isfile(dataset_path):
                artifact.add_file(dataset_path)
            elif os.path.isdir(dataset_path):
                artifact.add_dir(dataset_path)
            else:
                raise ValueError(f"Invalid dataset path: {dataset_path}")
            
            self.run.log_artifact(artifact)
            print(f"✓ Logged dataset artifact: {artifact_name}")
            
        except Exception as e:
            print(f"❌ Failed to log dataset: {e}")
    
    def log_config(self, config: Dict[str, Any]) -> None:
        """
        Update run configuration.
        
        Args:
            config: Configuration dictionary
        """
        if not self.enabled or self.run is None:
            return
        
        try:
            wandb.config.update(config)
        except Exception as e:
            print(f"❌ Failed to log config: {e}")
    
    def log_text(self, 
                 text: str,
                 key: str = "generated_text",
                 step: Optional[int] = None) -> None:
        """
        Log text samples to W&B.
        
        Args:
            text: Text to log
            key: Key for the text
            step: Step number
        """
        if not self.enabled or self.run is None:
            return
        
        try:
            wandb.log({key: wandb.Html(f"<pre>{text}</pre>")}, step=step)
        except Exception as e:
            print(f"❌ Failed to log text: {e}")
    
    def log_table(self,
                  data: List[List[Any]],
                  columns: List[str],
                  key: str = "results_table",
                  step: Optional[int] = None) -> None:
        """
        Log a data table to W&B.
        
        Args:
            data: List of rows (each row is a list of values)
            columns: Column names
            key: Key for the table
            step: Step number
        """
        if not self.enabled or self.run is None:
            return
        
        try:
            table = wandb.Table(data=data, columns=columns)
            wandb.log({key: table}, step=step)
        except Exception as e:
            print(f"❌ Failed to log table: {e}")
    
    def watch_model(self, 
                   model,
                   criterion=None,
                   log: str = "gradients",
                   log_freq: int = 100,
                   log_graph: bool = True) -> None:
        """
        Watch model parameters and gradients.
        
        Args:
            model: PyTorch model to watch
            criterion: Loss function
            log: What to log ("gradients", "parameters", "all", or None)
            log_freq: Logging frequency
            log_graph: Whether to log model graph
        """
        if not self.enabled or self.run is None:
            return
        
        try:
            wandb.watch(
                model,
                criterion=criterion,
                log=log,
                log_freq=log_freq,
                log_graph=log_graph
            )
            print("✓ Started watching model")
        except Exception as e:
            print(f"❌ Failed to watch model: {e}")
    
    def finish(self) -> None:
        """Finish the W&B run."""
        if not self.enabled or self.run is None:
            return
        
        try:
            wandb.finish()
            print("✓ W&B run finished")
        except Exception as e:
            print(f"❌ Failed to finish W&B run: {e}")
    
    def save_code(self, code_dir: str = ".") -> None:
        """
        Save code snapshot to W&B.
        
        Args:
            code_dir: Directory containing code to save
        """
        if not self.enabled or self.run is None:
            return
        
        try:
            wandb.run.log_code(code_dir)
            print(f"✓ Saved code from {code_dir}")
        except Exception as e:
            print(f"❌ Failed to save code: {e}")
    
    def alert(self, 
              title: str,
              text: str,
              level: str = "INFO",
              wait_duration: int = 300) -> None:
        """
        Send an alert notification.
        
        Args:
            title: Alert title
            text: Alert message
            level: Alert level ("INFO", "WARN", "ERROR")
            wait_duration: Minimum time between alerts
        """
        if not self.enabled or self.run is None:
            return
        
        try:
            wandb.alert(
                title=title,
                text=text,
                level=getattr(wandb.AlertLevel, level, wandb.AlertLevel.INFO),
                wait_duration=wait_duration
            )
            print(f"✓ Sent alert: {title}")
        except Exception as e:
            print(f"❌ Failed to send alert: {e}")


class WandbCallback:
    """
    Callback for automatic W&B logging during training.
    """
    
    def __init__(self, 
                 logger: JadioWandbLogger,
                 log_freq: int = 100,
                 log_model_freq: int = 1000,
                 log_text_freq: int = 500):
        """
        Initialize callback.
        
        Args:
            logger: W&B logger instance
            log_freq: Frequency for logging metrics
            log_model_freq: Frequency for logging model checkpoints
            log_text_freq: Frequency for logging text samples
        """
        self.logger = logger
        self.log_freq = log_freq
        self.log_model_freq = log_model_freq
        self.log_text_freq = log_text_freq
    
    def on_step_end(self, 
                   step: int,
                   metrics: Dict[str, float],
                   model_path: Optional[str] = None,
                   text_samples: Optional[List[str]] = None) -> None:
        """
        Called at the end of each training step.
        
        Args:
            step: Current step
            metrics: Metrics to log
            model_path: Path to model checkpoint (if saved)
            text_samples: Generated text samples
        """
        # Log metrics
        if step % self.log_freq == 0:
            self.logger.log(metrics, step=step)
        
        # Log model checkpoint
        if model_path and step % self.log_model_freq == 0:
            self.logger.log_model(
                model_path,
                name=f"checkpoint_step_{step}",
                aliases=["latest"] if step == self.log_model_freq else None
            )
        
        # Log text samples
        if text_samples and step % self.log_text_freq == 0:
            for i, text in enumerate(text_samples[:3]):  # Log first 3 samples
                self.logger.log_text(
                    text,
                    key=f"sample_{i}",
                    step=step
                )
    
    def on_epoch_end(self,
                    epoch: int,
                    epoch_metrics: Dict[str, float],
                    model_path: Optional[str] = None) -> None:
        """
        Called at the end of each epoch.
        
        Args:
            epoch: Current epoch
            epoch_metrics: Epoch-level metrics
            model_path: Path to model checkpoint
        """
        self.logger.log(epoch_metrics, step=epoch * 1000)  # Use epoch * 1000 as step
        
        if model_path:
            self.logger.log_model(
                model_path,
                name=f"epoch_{epoch}",
                aliases=["epoch_latest"]
            )


def create_wandb_logger(config: Dict[str, Any],
                       model_name: str = "jadio",
                       **kwargs) -> JadioWandbLogger:
    """
    Factory function to create a W&B logger with standard configuration.
    
    Args:
        config: Model/training configuration
        model_name: Name of the model
        **kwargs: Additional arguments for JadioWandbLogger
        
    Returns:
        Configured W&B logger
    """
    # Extract key info for run naming
    model_size = config.get('n_embd', 'unknown')
    timestamp = int(time.time())
    
    # Create run name
    run_name = f"{model_name}_{model_size}d_{timestamp}"
    
    # Create tags
    tags = ["jadio", "language-model"]
    if 'n_layer' in config:
        tags.append(f"{config['n_layer']}L")
    if 'n_embd' in config:
        tags.append(f"{config['n_embd']}H")
    
    return JadioWandbLogger(
        name=run_name,
        config=config,
        tags=tags,
        notes=f"Jadio LLM training - {model_size} hidden dimensions",
        **kwargs
    )


def main():
    """Test the W&B integration."""
    print("[Jadio] Testing Weights & Biases integration...")
    
    if not WANDB_AVAILABLE:
        print("❌ wandb not installed. Install with: pip install wandb")
        return
    
    print("⚠ This test will create a real W&B run. Make sure you're logged in.")
    print("Run 'wandb login' if you haven't already.")
    
    # Test configuration
    test_config = {
        "model_name": "jadio_test",
        "n_layer": 6,
        "n_embd": 384,
        "n_head": 6,
        "learning_rate": 1e-4,
        "batch_size": 4,
        "sequence_length": 128
    }
    
    # Create logger
    logger = create_wandb_logger(
        config=test_config,
        project="jadio-test",
        job_type="test"
    )
    
    if not logger.enabled:
        print("W&B logging disabled, skipping test")
        return
    
    print("✓ Created W&B logger")
    
    # Test metric logging
    print("--- Testing metric logging ---")
    for step in range(10):
        metrics = {
            "train_loss": 3.0 - step * 0.1,
            "learning_rate": 1e-4 * (0.99 ** step),
            "step": step
        }
        logger.log(metrics, step=step)
    
    print("✓ Logged training metrics")
    
    # Test text logging
    print("--- Testing text logging ---")
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Hello world! This is a test of the Jadio tokenizer.",
        "Once upon a time, in a land far far away..."
    ]
    
    for i, text in enumerate(sample_texts):
        logger.log_text(text, key=f"sample_{i}", step=i)
    
    print("✓ Logged text samples")
    
    # Test table logging
    print("--- Testing table logging ---")
    table_data = [
        ["Sample 1", 2.5, 0.8],
        ["Sample 2", 2.3, 0.85],
        ["Sample 3", 2.1, 0.9]
    ]
    logger.log_table(
        data=table_data,
        columns=["Text", "Loss", "Accuracy"],
        key="evaluation_results"
    )
    
    print("✓ Logged results table")
    
    # Test config update
    print("--- Testing config update ---")
    logger.log_config({
        "final_loss": 2.1,
        "training_time": 120.5,
        "converged": True
    })
    
    print("✓ Updated configuration")
    
    # Test creating temporary model file and logging it
    print("--- Testing model artifact logging ---")
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        # Create a dummy "model" file
        import torch
        torch.save({"dummy": "model"}, f.name)
        
        logger.log_model(
            f.name,
            name="test_model",
            aliases=["test"],
            metadata={
                "parameters": "6M",
                "architecture": "transformer",
                "test_model": True
            }
        )
        
        # Clean up
        os.unlink(f.name)
    
    print("✓ Logged model artifact")
    
    # Finish run
    logger.finish()
    print("✓ Finished W&B run")
    
    print("\n[Jadio] W&B integration test completed!")
    print("Check your W&B dashboard to see the logged data.")


if __name__ == "__main__":
    main()
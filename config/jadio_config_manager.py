"""
Configuration manager for the Jadio LLM.
"""
import json
import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
import copy

from .jadio_config import JadioConfig, get_jadio_50m_config, get_jadio_small_config


class JadioConfigManager:
    """
    Manager for Jadio LLM configurations.
    
    Handles loading, saving, and managing different configurations
    for training, evaluation, and generation.
    """
    
    def __init__(self, config_dir: str = "config"):
        """
        Initialize config manager.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache for loaded configs
        self._config_cache = {}
    
    def save_config(self, config: JadioConfig, name: str, overwrite: bool = False) -> str:
        """
        Save a configuration to file.
        
        Args:
            config: Configuration to save
            name: Name for the configuration
            overwrite: Whether to overwrite existing config
            
        Returns:
            Path to saved configuration file
        """
        config_path = self.config_dir / f"{name}.json"
        
        if config_path.exists() and not overwrite:
            raise FileExistsError(f"Configuration '{name}' already exists. Use overwrite=True to replace.")
        
        config_dict = config.to_dict()
        
        # Add metadata
        config_dict['_metadata'] = {
            'name': name,
            'config_type': 'jadio_model',
            'version': '1.0'
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"✓ Saved configuration '{name}' to {config_path}")
        return str(config_path)
    
    def load_config(self, name: str, use_cache: bool = True) -> JadioConfig:
        """
        Load a configuration from file.
        
        Args:
            name: Name of the configuration
            use_cache: Whether to use cached config
            
        Returns:
            Loaded configuration
        """
        if use_cache and name in self._config_cache:
            return copy.deepcopy(self._config_cache[name])
        
        config_path = self.config_dir / f"{name}.json"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration '{name}' not found at {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Remove metadata before creating config
        if '_metadata' in config_dict:
            del config_dict['_metadata']
        
        config = JadioConfig.from_dict(config_dict)
        
        if use_cache:
            self._config_cache[name] = copy.deepcopy(config)
        
        return config
    
    def list_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        List all available configurations.
        
        Returns:
            Dictionary mapping config names to their metadata
        """
        configs = {}
        
        for config_file in self.config_dir.glob("*.json"):
            name = config_file.stem
            
            try:
                with open(config_file, 'r') as f:
                    config_dict = json.load(f)
                
                metadata = config_dict.get('_metadata', {})
                metadata.update({
                    'file_path': str(config_file),
                    'n_layer': config_dict.get('n_layer', 'unknown'),
                    'n_embd': config_dict.get('n_embd', 'unknown'),
                    'n_head': config_dict.get('n_head', 'unknown'),
                    'vocab_size': config_dict.get('vocab_size', 'unknown'),
                })
                
                configs[name] = metadata
                
            except Exception as e:
                print(f"Warning: Could not load metadata for {config_file}: {e}")
        
        return configs
    
    def delete_config(self, name: str) -> bool:
        """
        Delete a configuration file.
        
        Args:
            name: Name of configuration to delete
            
        Returns:
            True if deleted successfully
        """
        config_path = self.config_dir / f"{name}.json"
        
        if not config_path.exists():
            print(f"Configuration '{name}' not found")
            return False
        
        config_path.unlink()
        
        # Remove from cache
        if name in self._config_cache:
            del self._config_cache[name]
        
        print(f"✓ Deleted configuration '{name}'")
        return True
    
    def create_variant(self, 
                      base_config: Union[str, JadioConfig], 
                      name: str,
                      modifications: Dict[str, Any]) -> JadioConfig:
        """
        Create a variant of an existing configuration.
        
        Args:
            base_config: Base configuration name or object
            name: Name for the new variant
            modifications: Dictionary of parameters to modify
            
        Returns:
            New configuration with modifications applied
        """
        if isinstance(base_config, str):
            config = self.load_config(base_config)
        else:
            config = copy.deepcopy(base_config)
        
        # Apply modifications
        config_dict = config.to_dict()
        config_dict.update(modifications)
        
        # Create new config
        new_config = JadioConfig.from_dict(config_dict)
        
        # Save the variant
        self.save_config(new_config, name, overwrite=True)
        
        return new_config
    
    def create_training_config(self, 
                             model_config: Union[str, JadioConfig],
                             training_params: Dict[str, Any],
                             name: str) -> Dict[str, Any]:
        """
        Create a complete training configuration.
        
        Args:
            model_config: Model configuration name or object
            training_params: Training parameters
            name: Name for the training config
            
        Returns:
            Complete training configuration
        """
        if isinstance(model_config, str):
            model_config = self.load_config(model_config)
        
        training_config = {
            'model_config': model_config.to_dict(),
            'training': {
                'learning_rate': 3e-4,
                'weight_decay': 0.1,
                'warmup_steps': 1000,
                'max_steps': 10000,
                'batch_size': 8,
                'gradient_clip': 1.0,
                'eval_interval': 500,
                'save_interval': 1000,
                **training_params
            },
            'optimizer': {
                'type': 'adamw',
                'betas': [0.9, 0.95],
                'eps': 1e-8
            },
            'scheduler': {
                'type': 'cosine',
                'min_lr_ratio': 0.1
            },
            '_metadata': {
                'name': name,
                'config_type': 'jadio_training',
                'version': '1.0'
            }
        }
        
        # Save training config
        config_path = self.config_dir / f"{name}_training.json"
        with open(config_path, 'w') as f:
            json.dump(training_config, f, indent=2)
        
        print(f"✓ Saved training configuration '{name}' to {config_path}")
        return training_config
    
    def load_training_config(self, name: str) -> Dict[str, Any]:
        """
        Load a training configuration.
        
        Args:
            name: Name of the training configuration
            
        Returns:
            Training configuration dictionary
        """
        config_path = self.config_dir / f"{name}_training.json"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Training configuration '{name}' not found")
        
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def create_generation_config(self, 
                               generation_params: Dict[str, Any],
                               name: str) -> Dict[str, Any]:
        """
        Create a generation configuration.
        
        Args:
            generation_params: Generation parameters
            name: Name for the generation config
            
        Returns:
            Generation configuration
        """
        generation_config = {
            'generation': {
                'max_new_tokens': 100,
                'temperature': 0.8,
                'top_k': 50,
                'top_p': 0.9,
                'do_sample': True,
                'num_return_sequences': 1,
                'repetition_penalty': 1.0,
                'length_penalty': 1.0,
                'early_stopping': True,
                **generation_params
            },
            '_metadata': {
                'name': name,
                'config_type': 'jadio_generation',
                'version': '1.0'
            }
        }
        
        # Save generation config
        config_path = self.config_dir / f"{name}_generation.json"
        with open(config_path, 'w') as f:
            json.dump(generation_config, f, indent=2)
        
        print(f"✓ Saved generation configuration '{name}' to {config_path}")
        return generation_config
    
    def initialize_default_configs(self) -> None:
        """Initialize default configurations."""
        print("Initializing default Jadio configurations...")
        
        # Save default model configs
        small_config = get_jadio_small_config()
        self.save_config(small_config, "small", overwrite=True)
        
        large_config = get_jadio_50m_config()
        self.save_config(large_config, "50m", overwrite=True)
        
        # Create training configs
        self.create_training_config(
            small_config,
            {
                'learning_rate': 1e-4,
                'batch_size': 4,
                'max_steps': 5000,
                'warmup_steps': 500
            },
            "small"
        )
        
        self.create_training_config(
            large_config,
            {
                'learning_rate': 3e-4,
                'batch_size': 8,
                'max_steps': 50000,
                'warmup_steps': 2000
            },
            "50m"
        )
        
        # Create generation configs
        self.create_generation_config(
            {
                'temperature': 0.7,
                'top_k': 40,
                'max_new_tokens': 50
            },
            "conservative"
        )
        
        self.create_generation_config(
            {
                'temperature': 1.0,
                'top_k': 100,
                'top_p': 0.95,
                'max_new_tokens': 200
            },
            "creative"
        )
        
        print("✓ Default configurations initialized")
    
    def export_config(self, name: str, format: str = 'json') -> str:
        """
        Export configuration to different formats.
        
        Args:
            name: Configuration name
            format: Export format ('json', 'yaml')
            
        Returns:
            Path to exported file
        """
        config = self.load_config(name)
        config_dict = config.to_dict()
        
        if format.lower() == 'json':
            export_path = self.config_dir / f"{name}_export.json"
            with open(export_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        
        elif format.lower() == 'yaml':
            try:
                export_path = self.config_dir / f"{name}_export.yaml"
                with open(export_path, 'w') as f:
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            except ImportError:
                raise ImportError("PyYAML required for YAML export. Install with: pip install pyyaml")
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        print(f"✓ Exported configuration '{name}' to {export_path}")
        return str(export_path)
    
    def validate_config(self, config: Union[str, JadioConfig]) -> Dict[str, Any]:
        """
        Validate a configuration.
        
        Args:
            config: Configuration name or object
            
        Returns:
            Validation results
        """
        if isinstance(config, str):
            config = self.load_config(config)
        
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'info': {}
        }
        
        # Basic validation
        try:
            # Check required parameters
            required_params = ['vocab_size', 'n_layer', 'n_embd', 'n_head', 'n_positions']
            for param in required_params:
                if not hasattr(config, param):
                    validation_results['errors'].append(f"Missing required parameter: {param}")
                    validation_results['valid'] = False
            
            # Check parameter ranges
            if config.n_embd % config.n_head != 0:
                validation_results['errors'].append(
                    f"n_embd ({config.n_embd}) must be divisible by n_head ({config.n_head})"
                )
                validation_results['valid'] = False
            
            # Check reasonable ranges
            if config.n_layer < 1 or config.n_layer > 100:
                validation_results['warnings'].append(
                    f"Unusual number of layers: {config.n_layer}"
                )
            
            if config.n_embd < 64 or config.n_embd > 8192:
                validation_results['warnings'].append(
                    f"Unusual embedding dimension: {config.n_embd}"
                )
            
            # Calculate model info
            validation_results['info'] = {
                'estimated_parameters': f"{config.get_model_params() / 1e6:.1f}M",
                'head_dimension': config.head_dim,
                'context_length': config.n_ctx,
                'vocabulary_size': config.vocab_size
            }
            
        except Exception as e:
            validation_results['errors'].append(f"Validation error: {str(e)}")
            validation_results['valid'] = False
        
        return validation_results


def main():
    """Test the configuration manager."""
    print("[Jadio] Testing Configuration Manager...")
    
    # Create config manager
    config_manager = JadioConfigManager("test_configs")
    print("✓ Created configuration manager")
    
    # Initialize default configs
    config_manager.initialize_default_configs()
    
    # List configurations
    print("\n--- Available Configurations ---")
    configs = config_manager.list_configs()
    for name, metadata in configs.items():
        print(f"  {name}: {metadata.get('n_layer', '?')}L, {metadata.get('n_embd', '?')}H")
    
    # Load and validate a config
    print("\n--- Testing Config Loading ---")
    small_config = config_manager.load_config("small")
    print(f"✓ Loaded small config: {small_config.n_layer}L, {small_config.n_embd}H")
    
    validation = config_manager.validate_config(small_config)
    print(f"Config validation: {'✓ Valid' if validation['valid'] else '❌ Invalid'}")
    if validation['errors']:
        for error in validation['errors']:
            print(f"  Error: {error}")
    if validation['warnings']:
        for warning in validation['warnings']:
            print(f"  Warning: {warning}")
    
    print("Info:")
    for key, value in validation['info'].items():
        print(f"  {key}: {value}")
    
    # Create a variant
    print("\n--- Testing Config Variants ---")
    variant_config = config_manager.create_variant(
        "small",
        "small_test",
        {
            'n_layer': 4,
            'learning_rate': 1e-5
        }
    )
    print(f"✓ Created variant: {variant_config.n_layer}L, {variant_config.n_embd}H")
    
    # Test training config
    print("\n--- Testing Training Configuration ---")
    training_config = config_manager.create_training_config(
        "small",
        {
            'learning_rate': 2e-4,
            'batch_size': 6,
            'max_steps': 1000
        },
        "test"
    )
    print("✓ Created training configuration")
    
    loaded_training = config_manager.load_training_config("test")
    print(f"Training LR: {loaded_training['training']['learning_rate']}")
    print(f"Training steps: {loaded_training['training']['max_steps']}")
    
    # Test generation config
    print("\n--- Testing Generation Configuration ---")
    config_manager.create_generation_config(
        {
            'temperature': 0.5,
            'max_new_tokens': 75
        },
        "test"
    )
    print("✓ Created generation configuration")
    
    # Test export
    print("\n--- Testing Config Export ---")
    try:
        export_path = config_manager.export_config("small", "json")
        print(f"✓ Exported to {export_path}")
    except Exception as e:
        print(f"❌ Export failed: {e}")
    
    # Clean up test configs
    print("\n--- Cleaning Up ---")
    import shutil
    shutil.rmtree("test_configs")
    print("✓ Cleaned up test configuration directory")
    
    print("\n[Jadio] Configuration manager test completed!")


if __name__ == "__main__":
    main()
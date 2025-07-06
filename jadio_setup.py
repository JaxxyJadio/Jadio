#!/usr/bin/env python3
"""
Setup script for the Jadio LLM project.
This script helps you get started with Jadio by setting up the environment
and running initial tests.
"""
import os
import sys
import subprocess
import argparse
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True


def install_dependencies():
    """Install required dependencies."""
    print("\n📦 Installing dependencies...")
    
    # Check if requirements.txt exists
    req_file = Path("requirements.txt")
    if req_file.exists():
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                         check=True)
            print("✓ Dependencies installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies")
            return False
    else:
        print("⚠ requirements.txt not found")
        return False


def check_gpu():
    """Check GPU availability."""
    print("\n🖥️ Checking GPU availability...")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✓ CUDA available: {gpu_count} GPU(s)")
            print(f"  Primary GPU: {gpu_name}")
            return True
        else:
            print("⚠ CUDA not available, will use CPU")
            return False
    except ImportError:
        print("❌ PyTorch not installed")
        return False


def check_tokenizer_files():
    """Check if GPT-2 tokenizer files are present."""
    print("\n🔤 Checking tokenizer files...")
    
    tokenizer_dir = Path("modelling/jadio01")
    required_files = ["vocab.json", "merges.txt", "tokenizer_config.json", "tokenizer.json"]
    
    missing_files = []
    for file in required_files:
        file_path = tokenizer_dir / file
        if file_path.exists():
            print(f"✓ Found {file}")
        else:
            missing_files.append(file)
            print(f"❌ Missing {file}")
    
    if missing_files:
        print(f"\n⚠ Missing {len(missing_files)} tokenizer files")
        print("These files should be the GPT-2 tokenizer files:")
        for file in missing_files:
            print(f"  - {tokenizer_dir / file}")
        print("\nYou can download them from HuggingFace:")
        print("  https://huggingface.co/gpt2/tree/main")
        return False
    else:
        print("✓ All tokenizer files present")
        return True


def run_tests():
    """Run basic tests to ensure everything works."""
    print("\n🧪 Running basic tests...")
    
    try:
        # Test imports
        print("Testing imports...")
        from config.jadio_config import get_jadio_small_config
        from modelling.jadio01.jadio_decoder_transformer import JadioLMHeadModel
        print("✓ Core modules import successfully")
        
        # Test model creation
        print("Testing model creation...")
        config = get_jadio_small_config()
        model = JadioLMHeadModel(config)
        print(f"✓ Model created: {model.get_num_params() / 1e6:.1f}M parameters")
        
        # Test forward pass
        print("Testing forward pass...")
        import torch
        input_ids = torch.randint(0, config.vocab_size, (1, 10))
        with torch.no_grad():
            output = model(input_ids)
        print(f"✓ Forward pass successful: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def run_integration_test():
    """Run the full integration test."""
    print("\n🔬 Running integration test...")
    try:
        from evaluation.jadio_test_integration import run_full_integration_test
        return run_full_integration_test()
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


def create_sample_training_script():
    """Create a sample training script."""
    print("\n📝 Creating sample training script...")
    
    script_content = '''#!/usr/bin/env python3
"""
Sample training script for Jadio LLM.
Run this to start training with dummy data.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from training.jadio_train import main

if __name__ == "__main__":
    # Override sys.argv to provide default arguments
    sys.argv = [
        "jadio_train.py",
        "--model_size", "small",
        "--batch_size", "4",
        "--num_epochs", "1",
        "--learning_rate", "1e-4",
        "--warmup_steps", "50",
        "--eval_interval", "50",
        "--save_interval", "100",
        "--dummy_data",
        "--seed", "42"
    ]
    main()
'''
    
    script_path = Path("train_sample.py")
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make executable on Unix systems
    if os.name != 'nt':
        os.chmod(script_path, 0o755)
    
    print(f"✓ Created sample training script: {script_path}")
    return script_path


def create_sample_generation_script():
    """Create a sample generation script."""
    print("📝 Creating sample generation script...")
    
    script_content = '''#!/usr/bin/env python3
"""
Sample generation script for Jadio LLM.
This will generate text once you have a trained model.
"""
import sys
import os
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    # Check if we have a trained model
    checkpoint_dir = Path("checkpoints")
    model_files = list(checkpoint_dir.glob("**/*.pt"))
    
    if not model_files:
        print("❌ No trained models found in checkpoints/")
        print("Train a model first with: python train_sample.py")
        return
    
    # Use the most recent model
    latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
    print(f"Using model: {latest_model}")
    
    # Import and run generator
    from scripts.jadio_generate import main as generate_main
    
    # Override sys.argv for generation
    sys.argv = [
        "jadio_generate.py",
        str(latest_model),
        "--interactive"
    ]
    
    generate_main()

if __name__ == "__main__":
    main()
'''
    
    script_path = Path("generate_sample.py")
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make executable on Unix systems
    if os.name != 'nt':
        os.chmod(script_path, 0o755)
    
    print(f"✓ Created sample generation script: {script_path}")
    return script_path


def print_usage_guide():
    """Print usage guide."""
    print("\n" + "=" * 60)
    print("🎯 JADIO LLM QUICK START GUIDE")
    print("=" * 60)
    
    print("\n1. 🏃 Quick Test (start here):")
    print("   python setup_jadio.py --test")
    
    print("\n2. 🚀 Train a Small Model:")
    print("   python train_sample.py")
    
    print("\n3. 💬 Generate Text:")
    print("   python generate_sample.py")
    
    print("\n4. 🔬 Run Full Tests:")
    print("   python evaluation/jadio_integration_test.py")
    
    print("\n5. 📚 Manual Training:")
    print("   python training/jadio_train.py --model_size small --dummy_data")
    
    print("\n6. 🎨 Interactive Generation:")
    print("   python scripts/jadio_generate.py <model_path> --interactive")
    
    print("\n📁 Important Directories:")
    print("   - checkpoints/    : Saved model checkpoints")
    print("   - logs/          : Training logs")
    print("   - data/          : Training data (add your own)")
    print("   - modelling/     : Model architecture")
    print("   - tokenizer/     : Tokenizer files")
    
    print("\n⚙️ Configuration:")
    print("   - Edit config/jadio_config.py for model settings")
    print("   - Small model: ~6M parameters (for testing)")
    print("   - 50M model: ~50M parameters (default)")
    
    print("\n🆘 Troubleshooting:")
    print("   - Missing tokenizer files? Download GPT-2 files from HuggingFace")
    print("   - CUDA issues? Add --device cpu to force CPU usage")
    print("   - Import errors? Check Python path and dependencies")


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Setup Jadio LLM project")
    parser.add_argument("--test", action="store_true", 
                       help="Run basic tests only")
    parser.add_argument("--full-test", action="store_true",
                       help="Run full integration test")
    parser.add_argument("--install", action="store_true",
                       help="Install dependencies")
    parser.add_argument("--create-scripts", action="store_true",
                       help="Create sample scripts")
    parser.add_argument("--all", action="store_true",
                       help="Run all setup steps")
    
    args = parser.parse_args()
    
    print("🚀 JADIO LLM SETUP")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        return 1
    
    success = True
    
    # Install dependencies if requested
    if args.install or args.all:
        if not install_dependencies():
            success = False
    
    # Check GPU
    check_gpu()
    
    # Check tokenizer files
    tokenizer_ok = check_tokenizer_files()
    if not tokenizer_ok:
        print("\n⚠ Warning: Some tokenizer files are missing")
        print("The basic tests may still work with fallback tokenizers")
    
    # Run tests
    if args.test or args.all:
        if not run_tests():
            success = False
    
    # Run full integration test
    if args.full_test or args.all:
        if not run_integration_test():
            success = False
    
    # Create sample scripts
    if args.create_scripts or args.all:
        create_sample_training_script()
        create_sample_generation_script()
    
    # Print usage guide
    if args.all or not any([args.test, args.full_test, args.install, args.create_scripts]):
        print_usage_guide()
    
    # Final status
    print("\n" + "=" * 60)
    if success:
        print("✅ SETUP COMPLETED SUCCESSFULLY!")
        print("\nJadio LLM is ready to use!")
        
        if not tokenizer_ok:
            print("\n⚠ Note: For full functionality, add GPT-2 tokenizer files to:")
            print("   modelling/jadio01/")
        
        print("\nTry: python setup_jadio.py --test")
    else:
        print("❌ SETUP ENCOUNTERED ISSUES")
        print("\nSome components may not work correctly.")
        print("Check the error messages above for details.")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
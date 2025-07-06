"""
model_loader.py with Updated Inline Documentation

This module is responsible for loading and unloading the Qwen language model and its tokenizer for the Jadio-quick-llm project.
It provides a simple, robust interface for other modules (main.py, chat_manager.py) to access the model and tokenizer, ensuring that only one model is loaded in memory at a time.

Design assumptions:
- This is a LAN-only, personal-use project (see README.md and flowguide.md).
- All inference is performed on the server (PC 2), never on the client.
- Model loading is lazy: the model is only loaded when first needed, and can be unloaded to free memory.
- Security is not a concern here, as only trusted users on the LAN can access the server.

Interactions:
- main.py calls ModelLoader to get the model and tokenizer for inference.
- chat_manager.py may also interact with ModelLoader for advanced agent tool use.
- config.py provides the MODEL_PATH setting.

If you are reading this in the future, remember: this file is designed for simplicity, maintainability, and easy debugging in a trusted environment.
"""

import os  # For checking if the model directory exists
from typing import Tuple, Optional  # For type hints and clarity
import sys  # For clean exit in CLI test

import torch  # For device and dtype selection, and freeing GPU memory
from transformers import AutoModelForCausalLM, AutoTokenizer  # Hugging Face model/tokenizer loading
from .config import MODEL_PATH, TORCH_DTYPE  # Path to the model directory and torch dtype, set in config.py

class ModelLoader:
    """
    Handles lazy loading and unloading of the Qwen model and tokenizer.
    Ensures only one model is loaded at a time to save memory.
    Provides a simple interface for main.py and chat_manager.py to get the model/tokenizer.
    Note: torch_dtype is selected automatically (float16 for CUDA, float32 otherwise).
    Model requirements: Model must implement 'generate', tokenizer must implement '__call__' and 'decode'.
    Only tested with HuggingFace-compatible Qwen and Llama models.
    """
    def __init__(self):
        # The loaded model (None if not loaded)
        self.model: Optional[AutoModelForCausalLM] = None
        # The loaded tokenizer (None if not loaded)
        self.tokenizer: Optional[AutoTokenizer] = None
        # Path to the model directory (from config.py)
        self.model_path = MODEL_PATH

    def is_loaded(self) -> bool:
        """
        Returns True if both model and tokenizer are loaded in memory.
        Used by main.py to check model status for /api/status.
        """
        return self.model is not None and self.tokenizer is not None

    def load(self):
        """
        Loads the model and tokenizer from disk if not already loaded.
        - Uses torch_dtype from config if set, else float16 if CUDA is available, else float32.
        - Uses device_map="auto" to let Hugging Face place the model on the best device.
        - Raises FileNotFoundError if the model directory is missing.
        - On failure, sets model/tokenizer to None and raises RuntimeError with full path info.
        """
        if self.is_loaded():
            return  # Already loaded, do nothing
        if not os.path.isdir(self.model_path):
            raise FileNotFoundError(f"Model directory not found: {self.model_path}")
        try:
            # Load the tokenizer from the model directory
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            # Determine torch dtype
            if TORCH_DTYPE:
                dtype = getattr(torch, TORCH_DTYPE)
            else:
                dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            # Load the model with appropriate dtype and device mapping
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=dtype,
                device_map="auto"
            )
            # --- Compatibility check ---
            required_tokenizer_methods = ["__call__", "decode"]
            for method in required_tokenizer_methods:
                if not hasattr(self.tokenizer, method):
                    raise RuntimeError(f"Tokenizer missing required method: {method}")
            required_model_methods = ["generate"]
            for method in required_model_methods:
                if not hasattr(self.model, method):
                    raise RuntimeError(f"Model missing required method: {method}")
        except Exception as e:
            # On failure, clear model/tokenizer and raise a clear error with path info
            self.model = None
            self.tokenizer = None
            raise RuntimeError(f"Failed to load model from {self.model_path}: {e}")

    def unload(self):
        """
        Unloads the model and tokenizer from memory.
        - Frees GPU memory by calling torch.cuda.empty_cache().
        - Used for manual memory management or when switching models.
        - Safe to call even if model/tokenizer are already None.
        - Note: empty_cache() is a no-op if CUDA is not available or nothing is loaded.
        """
        self.model = None
        self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_model_and_tokenizer(self) -> Tuple[Optional[AutoModelForCausalLM], Optional[AutoTokenizer]]:
        """
        Returns the loaded model and tokenizer, loading them if necessary.
        - Used by main.py for every inference request.
        - Ensures lazy loading: model is only loaded when first needed.
        - Returns (model, tokenizer) tuple (may be None if loading fails).
        - Raises RuntimeError if loading fails.
        """
        if not self.is_loaded():
            self.load()
        return self.model, self.tokenizer

# --- Unit tests for ModelLoader ---
# These tests can be run directly (python model_loader.py) to verify loading/unloading works.
if __name__ == "__main__":
    if not os.path.isdir(MODEL_PATH):
        print(f"Model directory not found at {MODEL_PATH}. Skipping test.")
        sys.exit(0)
    loader = ModelLoader()
    assert not loader.is_loaded(), "Model should not be loaded initially."
    try:
        loader.load()
        assert loader.is_loaded(), "Model should be loaded after calling load()."
        model, tokenizer = loader.get_model_and_tokenizer()
        assert model is not None and tokenizer is not None, "Model and tokenizer should not be None."
        loader.unload()
        assert not loader.is_loaded(), "Model should be unloaded after calling unload()."
        print("ModelLoader tests passed.")
    except Exception as e:
        print(f"ModelLoader test failed: {e}")

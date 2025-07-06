import logging
import random
import torch
import numpy as np
import os

def setup_logger(name="jadio", level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        ch = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_checkpoint(model, optimizer, scheduler, path):
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    torch.save(state, path)

def load_checkpoint(path, model, optimizer=None, scheduler=None):
    state = torch.load(path)
    model.load_state_dict(state["model_state_dict"])
    if optimizer and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    if scheduler and "scheduler_state_dict" in state:
        scheduler.load_state_dict(state["scheduler_state_dict"])

"""
Utility script for the Jadio LLM project.
"""
from scripts import jadio_utilities

def main():
    print("[Jadio] Utilities script stub.")

if __name__ == "__main__":
    main()

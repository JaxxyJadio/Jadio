# scripts/jadio_wandb.py
"""
Weights & Biases integration script for the Jadio LLM project.
"""
try:
    import wandb
except ImportError:
    wandb = None

def main():
    print("[Jadio] Weights & Biases script stub.")
    if wandb is None:
        print("wandb not installed.")
    else:
        print("wandb available.")

if __name__ == "__main__":
    main()

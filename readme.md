# Jadio

Jadio is a private large language model (LLM) project. This repository contains all code, scripts, and configuration for developing, training, and evaluating the Jadio LLM.

## Project Tree
```
Jadio/
├── checkpoints/
│   └── __init__.py
├── config/
│   ├── __init__.py
│   ├── jadio_config_manager.py
│   ├── jadio_config.py
│   ├── jadio_test_config.py
│   └── jadio_train_config.py
├── data/
│   ├── __init__.py
│   ├── jadio_dataset_loader.py
│   └── jadio_test_dataset_loader.py
├── evaluation/
│   ├── __init__.py
│   ├── jadio_benchmark_test.py
│   └── jadio_evaluation.py
├── metrics/
│   ├── __init__.py
│   └── jadio_metrics.py
├── modelling/
│   ├── __init__.py
│   ├── jadio_attention.py
│   ├── jadio_decoder_transformer.py
│   ├── jadio_embeddings.py
│   ├── jadio_feed_forward.py
│   └── jadio_layer_norm.py
├── scripts/
│   ├── __init__.py
│   ├── jadio_eval.py
│   ├── jadio_generate.py
│   ├── jadio_test.py
│   ├── jadio_train.py
│   ├── jadio_utilities.py
│   └── jadio_wandb.py
├── tokenizer/
│   ├── __init__.py
│   ├── jadio_test_tokenizer.py
│   ├── jadio_tokenizer.py
│   └── jadio_train_tokenizer.py
├── training/
│   ├── __init__.py
│   ├── jadio_optimizer.py
│   ├── jadio_scheduler.py
│   └── jadio_trainer.py
├── __init__.py
├── pyproject.toml
├── readme.md
├── requirements.txt
└── setup.py
```

## Notes
- This repository is private and intended for personal or internal use only.
- All code, data, and experiments are specific to the Jadio LLM project.


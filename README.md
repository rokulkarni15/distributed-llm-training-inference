# Distributed LLM Training & Inference

End-to-end pipeline for fine-tuning and deploying large language models at scale.

## Overview

This project demonstrates distributed training and optimized inference deployment of Llama 2 7B:

- **Training**: Multi-GPU fine-tuning using DeepSpeed ZeRO-3 with LoRA
- **Inference**: High-throughput serving with vLLM and tensor parallelism
- **Benchmarking**: Comprehensive performance analysis under load

## Tech Stack

- **Training**: PyTorch, DeepSpeed, PEFT (LoRA), Transformers
- **Inference**: vLLM, OpenAI API
- **Benchmarking**: Locust, AsyncIO
- **Orchestration**: SLURM









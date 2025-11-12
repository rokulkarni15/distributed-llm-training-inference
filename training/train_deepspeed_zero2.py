#!/usr/bin/env python3
"""
DeepSpeed ZeRO-2 training: PyTorch + LoRA + DeepSpeed ZeRO-2 on 1-4 GPUs.

Usage:
    # Single GPU
    python training/train_deepspeed_zero2.py --dataset_path ./data/glaive_code_full --resume_from_checkpoint
    
    # Multi-GPU (2, 3, or 4 GPUs)
    torchrun --nproc_per_node=2 training/train_deepspeed_zero2.py --dataset_path ./data/glaive_code_full --resume_from_checkpoint
    torchrun --nproc_per_node=3 training/train_deepspeed_zero2.py --dataset_path ./data/glaive_code_full --resume_from_checkpoint
    torchrun --nproc_per_node=4 training/train_deepspeed_zero2.py --dataset_path ./data/glaive_code_full --resume_from_checkpoint
"""

import os
import argparse
import time
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType

from training.utils import (
    save_training_metrics,
    print_metrics_summary,
    create_experiment_name,
    get_zero_stage_from_config
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune Llama 2 7B with LoRA + DeepSpeed ZeRO-2 (1-4 GPUs)"
    )
    
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="Base model to fine-tune"
    )
    
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./data/glaive_code_full",
        help="Path to preprocessed dataset"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save checkpoints (auto-generated if not provided)"
    )
    
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size per GPU"
    )
    
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=16,
        help="Gradient accumulation steps"
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate"
    )
    
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="LoRA rank"
    )
    
    # DeepSpeed config argument
    parser.add_argument(
        "--deepspeed_config",
        type=str,
        default="./training/ds_config_zero2.json",
        help="Path to DeepSpeed config file"
    )
    
    parser.add_argument(
        "--resume_from_checkpoint",
        action="store_true",
        help="Resume from last checkpoint if available"
    )
    
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training (automatically set by torchrun)"
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Detect distributed setup
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = world_size > 1
    is_main_process = local_rank in [-1, 0]
    
    # Detect ZeRO stage and create experiment name
    zero_stage = get_zero_stage_from_config(args.deepspeed_config)
    experiment_name = create_experiment_name(world_size, zero_stage)
    
    # Set output directory based on experiment
    if args.output_dir is None:
        args.output_dir = f"./checkpoints/{experiment_name}"
    
    # Only print from main process
    if is_main_process:
        print("\n" + "="*70)
        print(f"DEEPSPEED ZeRO-{zero_stage} TRAINING")
        print("="*70)
        print(f"\nExperiment: {experiment_name}")
        print(f"GPUs: {world_size}")
        print(f"ZeRO Stage: {zero_stage}")
        print(f"Config: {args.deepspeed_config}")
        print(f"Output: {args.output_dir}")
        print()
        print(f"NOTE: Using DeepSpeed ZeRO-{zero_stage} for optimizer + gradient state partitioning")
        if is_distributed:
            print(f"Expected: Greater memory savings + {world_size}x speedup with {world_size} GPUs")
        else:
            print("Expected: Lower memory usage than ZeRO-1, similar speed on 1 GPU")
        print()
    
    # Check GPU
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    
    if is_main_process:
        if is_distributed:
            print(f"Distributed Training: {world_size} GPUs")
            for i in range(world_size):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print(f"Single GPU Training")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load tokenizer
    if is_main_process:
        print("\n[1/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if is_main_process:
        print("Tokenizer loaded")
    
    # Load model
    if is_main_process:
        print("\n[2/5] Loading model...")
    
    # For distributed training, load model differently
    if is_distributed:
        # Don't use device_map="auto" for multi-GPU, let DeepSpeed handle it
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.float16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    
    if is_main_process:
        print("Model loaded")
    
    # Apply LoRA
    if is_main_process:
        print(f"\n[3/5] Applying LoRA (r={args.lora_r})...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    
    model = get_peft_model(model, lora_config)
    if is_main_process:
        model.print_trainable_parameters()
        print("LoRA applied")
    
    # Load dataset
    if is_main_process:
        print("\n[4/5] Loading dataset...")
    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset not found: {args.dataset_path}")
    
    dataset = load_from_disk(args.dataset_path)
    
    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding=False,
        )
    
    tokenized = dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing" if is_main_process else None,
    )
    
    if is_main_process:
        print(f"Dataset ready: {len(tokenized):,} samples")
    
    # Training setup with DeepSpeed
    if is_main_process:
        print("\n[5/5] Configuring training with DeepSpeed...")
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        
        # DeepSpeed integration
        deepspeed=args.deepspeed_config,
        
        # Memory optimization (DeepSpeed will handle fp16)
        gradient_checkpointing=True,
        
        # Logging
        logging_steps=10,
        logging_dir=f"{args.output_dir}/logs",
        
        # Saving - Save more frequently for cluster resilience
        save_strategy="steps",
        save_steps=100,  # Save every 100 steps
        save_total_limit=3,  # Keep last 3 checkpoints
        
        # Distributed training settings
        ddp_find_unused_parameters=False,
        
        # Disable reporting
        report_to="none",
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )
    
    # Check for existing checkpoint
    resume_checkpoint = None
    if args.resume_from_checkpoint and is_main_process:
        if os.path.exists(args.output_dir):
            checkpoints = [d for d in os.listdir(args.output_dir) 
                          if d.startswith("checkpoint-") and os.path.isdir(os.path.join(args.output_dir, d))]
            if checkpoints:
                # Get the latest checkpoint
                latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[1]))
                resume_checkpoint = os.path.join(args.output_dir, latest_checkpoint)
                print(f"\n✓ Found checkpoint: {resume_checkpoint}")
                print("  Training will resume from this checkpoint")
            else:
                print("\n✓ No checkpoint found, starting fresh training")
    
    if is_main_process:
        print("✓ Trainer configured with DeepSpeed ZeRO-2")
        total_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps * world_size
        print(f"\nEffective batch size: {total_batch_size}")
        print(f"  = {args.per_device_train_batch_size} (per_device) × {args.gradient_accumulation_steps} (grad_accum) × {world_size} (GPUs)")
    
    # Train
    if is_main_process:
        print("\n" + "="*70)
        print(f"STARTING TRAINING: {experiment_name.upper()}")
        print("="*70)
        print()
    
    start_time = time.time()
    
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    
    training_time = time.time() - start_time
    
    # Save and collect metrics (only main process)
    if is_main_process:
        print("\n" + "="*70)
        print("SAVING MODEL")
        print("="*70)
        
        final_dir = f"{args.output_dir}/final"
        trainer.save_model(final_dir)
        tokenizer.save_pretrained(final_dir)
        
        print(f"✓ Model saved to {final_dir}")
        
        # Collect metrics
        print("\n" + "="*70)
        print("COLLECTING METRICS")
        print("="*70)
        
        num_samples = len(tokenized)
        
        # Get final loss from training history
        train_history = trainer.state.log_history
        final_loss = None
        for entry in reversed(train_history):
            if 'loss' in entry:
                final_loss = entry['loss']
                break
        
        # Prepare metrics dictionary
        metrics = {
            "experiment": experiment_name,
            "num_gpus": world_size,
            "zero_stage": zero_stage,
            "strategy": f"deepspeed_zero{zero_stage}",
            "training_time_hours": training_time / 3600,
            "samples_per_second": (num_samples * args.num_train_epochs) / training_time,
            "peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9,
            "final_loss": final_loss if final_loss is not None else 0.0,
        }
        
        # Print and save metrics
        print_metrics_summary(metrics)
        save_training_metrics(metrics)
        
        # Final summary
        print("\n" + "="*70)
        print(f"{experiment_name.upper()} TRAINING COMPLETE!")
        print("="*70)
        print(f"\nTime: {training_time/3600:.2f} hours")
        print(f"Throughput: {metrics['samples_per_second']:.1f} samples/sec")
        print(f"Memory: {metrics['peak_memory_gb']:.2f} GB")
        print(f"\nCheckpoints: {args.output_dir}")
        print(f"Metrics: results/training_metrics.csv")
        print("\nRun 'python scripts/compare_training.py' to see comparison")
        print()


if __name__ == "__main__":
    main()
#!/usr/bin/env python
"""
DeepSpeed ZeRO-3 training: Works with 1, 2, or 4 GPUs

Usage:
    # 1 GPU
    deepspeed --num_gpus=1 training/train_zero3.py --deepspeed configs/ds_config_zero3.json
    
    # 4 GPUs
    deepspeed --num_gpus=4 training/train_zero3.py --deepspeed configs/ds_config_zero3.json
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
        description="Fine-tune with DeepSpeed ZeRO-3 (flexible GPU count)"
    )
    
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="Base model"
    )
    
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./data/glaive_code_full",
        help="Preprocessed dataset path"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Checkpoint directory (auto-generated if not provided)"
    )
    
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Number of epochs"
    )
    
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=2,
        help="Batch size per GPU"
    )
    
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation"
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
    
    parser.add_argument(
        "--deepspeed",
        type=str,
        required=True,
        help="DeepSpeed config file"
    )
    
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training"
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Detect GPU count and ZeRO stage
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    zero_stage = get_zero_stage_from_config(args.deepspeed)
    experiment_name = create_experiment_name(num_gpus, zero_stage)
    
    # Set output directory based on experiment
    if args.output_dir is None:
        args.output_dir = f"./checkpoints/{experiment_name}"
    
    # Only print on main process
    if args.local_rank <= 0:
        print("\n" + "="*70)
        print(f"DEEPSPEED ZeRO-{zero_stage} TRAINING")
        print("="*70)
        print(f"\nExperiment: {experiment_name}")
        print(f"GPUs: {num_gpus}")
        print(f"ZeRO Stage: {zero_stage}")
        print(f"Config: {args.deepspeed}")
        print(f"Output: {args.output_dir}")
        print()
        
        if num_gpus == 1:
            print("NOTE: Running on 1 GPU with CPU offloading for memory efficiency")
        else:
            print(f"NOTE: Running on {num_gpus} GPUs with parameter sharding + CPU offloading")
        print()
    
    # Check CUDA
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    
    # Load tokenizer
    if args.local_rank <= 0:
        print("[1/5] Loading tokenizer...")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if args.local_rank <= 0:
        print("Tokenizer loaded")
    
    # Load model
    if args.local_rank <= 0:
        print("[2/5] Loading model...")
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
    )
    
    if args.local_rank <= 0:
        print("Model loaded")
    
    # Apply LoRA
    if args.local_rank <= 0:
        print(f"[3/5] Applying LoRA (r={args.lora_r})...")
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    
    model = get_peft_model(model, lora_config)
    
    if args.local_rank <= 0:
        model.print_trainable_parameters()
        print("LoRA applied")
    
    # Load dataset
    if args.local_rank <= 0:
        print("[4/5] Loading dataset...")
    
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
        desc="Tokenizing" if args.local_rank <= 0 else None,
    )
    
    if args.local_rank <= 0:
        print(f"Dataset ready: {len(tokenized):,} samples")
    
    # Training setup
    if args.local_rank <= 0:
        print("[5/5] Configuring training with DeepSpeed ZeRO-3...")
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        # Optimization
        fp16=True,
        gradient_checkpointing=True,
        # Logging
        logging_steps=10,
        logging_dir=f"{args.output_dir}/logs",
        # Saving
        save_strategy="epoch",
        save_total_limit=2,
        # DeepSpeed
        deepspeed=args.deepspeed,
        # Distributed
        local_rank=args.local_rank,
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
    
    if args.local_rank <= 0:
        print("✓ Trainer configured")
        effective_bs = args.per_device_train_batch_size * args.gradient_accumulation_steps * num_gpus
        print(f"Effective batch size: {effective_bs}")
    
    # Train
    if args.local_rank <= 0:
        print("\n" + "="*70)
        print(f"STARTING TRAINING: {experiment_name.upper()}")
        print("="*70)
        print()
    
    start_time = time.time()
    
    trainer.train()
    
    training_time = time.time() - start_time
    
    # Save and collect metrics
    if args.local_rank <= 0:
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
        
        # Get final loss
        train_history = trainer.state.log_history
        final_loss = None
        for entry in reversed(train_history):
            if 'loss' in entry:
                final_loss = entry['loss']
                break
        
        # Metrics
        metrics = {
            "experiment": experiment_name,
            "num_gpus": num_gpus,
            "zero_stage": zero_stage,
            "strategy": f"deepspeed_zero{zero_stage}",
            "training_time_hours": training_time / 3600,
            "samples_per_second": (num_samples * args.num_train_epochs) / training_time,
            "peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9,
            "final_loss": final_loss if final_loss is not None else 0.0,
        }
        
        # Print and save
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
#!/usr/bin/env python3
"""
DeepSpeed ZeRO-2 training: PyTorch + LoRA + DeepSpeed ZeRO-2 on 1-4 GPUs.

Usage:
    # Single GPU
    deepspeed --num_gpus=1 training/train_deepspeed_zero2.py --dataset_path ./data/glaive_code_full --resume_from_checkpoint
    
    # Multi-GPU (2, 3, or 4 GPUs)
    deepspeed --num_gpus=2 training/train_deepspeed_zero2.py --dataset_path ./data/glaive_code_full --resume_from_checkpoint
    deepspeed --num_gpus=3 training/train_deepspeed_zero2.py --dataset_path ./data/glaive_code_full --resume_from_checkpoint
    deepspeed --num_gpus=4 training/train_deepspeed_zero2.py --dataset_path ./data/glaive_code_full --resume_from_checkpoint
"""

import os
import json
import argparse
import time
import torch
import pandas as pd
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType

from torch.utils.data import DataLoader

def get_dataloader(tokenized, batch_size, world_size):
    return DataLoader(
        tokenized,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,  # Parallel data loading
        pin_memory=True,  # Faster GPU transfer
        persistent_workers=True,  # Better multi-epoch performance
    )


def get_zero_stage_from_config(config_path):
    """Extract ZeRO stage from DeepSpeed config file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config.get('zero_optimization', {}).get('stage', 0)
    except Exception as e:
        print(f"Warning: Could not read ZeRO stage from config: {e}")
        return 0


def create_experiment_name(num_gpus, zero_stage):
    """Create consistent experiment name."""
    return f"zero{zero_stage}_{num_gpus}gpu"


def print_metrics_summary(metrics):
    """Print metrics in a formatted way."""
    print("\n" + "="*70)
    print("TRAINING METRICS SUMMARY")
    print("="*70)
    print(f"Experiment:              {metrics['experiment']}")
    print(f"Strategy:                {metrics['strategy']}")
    print(f"Number of GPUs:          {metrics['num_gpus']}")
    print(f"ZeRO Stage:              {metrics['zero_stage']}")
    print(f"Session Time:            {metrics['training_time_hours']:.4f} hours")
    print(f"Cumulative Time:         {metrics['cumulative_time_hours']:.4f} hours")
    print(f"Total Steps:             {metrics['total_steps']:,}")
    print(f"Session Samples:         {metrics['samples_processed']:,}")
    print(f"Session Throughput:      {metrics['samples_per_second']:.2f} samples/sec")
    print(f"Cumulative Throughput:   {metrics['cumulative_samples_per_second']:.2f} samples/sec")
    print(f"Peak Memory:             {metrics['peak_memory_gb']:.2f} GB")
    print(f"Final Loss:              {metrics['final_loss']:.4f}")
    print(f"Target Epochs:           {metrics['target_epochs']:.2f}")
    print(f"Actual Epochs:           {metrics['actual_epochs']:.2f}")
    print("="*70)


def save_training_metrics(metrics, results_dir="results"):
    """
    Save metrics to separate CSV files for each configuration.
    
    File naming: results/zero{stage}_{num_gpus}gpu_metrics.csv
    Example: results/zero2_1gpu_metrics.csv, results/zero2_2gpu_metrics.csv
    """
    os.makedirs(results_dir, exist_ok=True)
    
    # Create filename based on experiment configuration
    experiment_name = metrics['experiment']
    csv_filename = f"{experiment_name}_metrics.csv"
    csv_path = os.path.join(results_dir, csv_filename)
    
    # Create DataFrame from metrics
    df_new = pd.DataFrame([metrics])
    
    # Append to existing CSV or create new one
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        try:
            df_existing = pd.read_csv(csv_path)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.to_csv(csv_path, index=False)
            print(f"\n✓ Metrics appended to: {csv_path}")
        except (pd.errors.EmptyDataError, Exception) as e:
            # If file is corrupted or empty, overwrite it
            print(f"Warning: Could not read existing file ({e}), creating new file")
            df_new.to_csv(csv_path, index=False)
            print(f"✓ Metrics saved to: {csv_path}")
    else:
        df_new.to_csv(csv_path, index=False)
        print(f"\n✓ Metrics saved to: {csv_path}")
    
    # Also save to a combined file for easy comparison
    combined_csv_path = os.path.join(results_dir, "all_experiments_metrics.csv")
    if os.path.exists(combined_csv_path) and os.path.getsize(combined_csv_path) > 0:
        try:
            df_all = pd.read_csv(combined_csv_path)
            df_all = pd.concat([df_all, df_new], ignore_index=True)
            df_all.to_csv(combined_csv_path, index=False)
            print(f"✓ Metrics also appended to combined file: {combined_csv_path}")
        except (pd.errors.EmptyDataError, Exception) as e:
            # If file is corrupted or empty, overwrite it
            print(f"Warning: Could not read combined file ({e}), creating new file")
            df_new.to_csv(combined_csv_path, index=False)
            print(f"✓ Metrics saved to combined file: {combined_csv_path}")
    else:
        df_new.to_csv(combined_csv_path, index=False)
        print(f"✓ Metrics also saved to combined file: {combined_csv_path}")


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
        type=float,
        default=None,
        help="Number of training epochs (if provided, overrides target_total_steps)"
    )
    
    parser.add_argument(
        "--target_total_steps",
        type=int,
        default=10000,
        help="Target total training steps (used to auto-calculate epochs if num_train_epochs is None)"
    )
    
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="Batch size per GPU"
    )
    
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
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
    
    parser.add_argument(
        "--deepspeed_config",
        type=str,
        default="../configs/ds_config_zero2.json",
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
        help="Local rank for distributed training (automatically set by DeepSpeed)"
    )
    
    return parser.parse_args()


def load_cumulative_metrics(output_dir, results_dir, experiment_name):
    """Load cumulative training time and steps from the experiment-specific CSV file."""
    cumulative_time = 0.0
    cumulative_global_steps = 0
    
    # Load from experiment-specific CSV if exists
    csv_filename = f"{experiment_name}_metrics.csv"
    csv_path = os.path.join(results_dir, csv_filename)
    
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        try:
            df = pd.read_csv(csv_path)
            if not df.empty:
                # Sum up all previous sessions for this experiment
                cumulative_time = df['training_time_hours'].sum()
                cumulative_global_steps = df['total_steps'].max()
                return cumulative_time, cumulative_global_steps
        except (pd.errors.EmptyDataError, Exception) as e:
            print(f"Warning: Could not load cumulative metrics from CSV: {e}")
    
    # Fallback: Try to load from last checkpoint
    if os.path.exists(output_dir):
        checkpoints = [d for d in os.listdir(output_dir) 
                      if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))]
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[1]))
            checkpoint_step = int(latest_checkpoint.split("-")[1])
            cumulative_global_steps = checkpoint_step
            print(f"Loaded steps from checkpoint: {cumulative_global_steps}")
    
    return cumulative_time, cumulative_global_steps


def calculate_epochs(args, num_samples, world_size):
    """Calculate epochs based on target steps or provided epochs."""
    if args.num_train_epochs is not None:
        return args.num_train_epochs
    
    effective_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps * world_size
    steps_per_epoch = num_samples / effective_batch_size
    epochs = args.target_total_steps / steps_per_epoch
    return epochs


def main():
    """Main training function."""
    # Set environment variables BEFORE any CUDA operations
    os.environ["DS_BUILD_OPS"] = "0"
    os.environ["DS_BUILD_CPU_ADAM"] = "0"
    os.environ["DS_BUILD_FUSED_ADAM"] = "1"
    os.environ["DS_BUILD_UTILS"] = "0"
    os.environ["TORCH_CUDA_ARCH_LIST"] = "7.0"
    
    args = parse_args()
    
    # Detect distributed setup - DeepSpeed sets these automatically
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = world_size > 1
    is_main_process = local_rank in [-1, 0]
    
    # Set CUDA device for this process
    if local_rank != -1:
        torch.cuda.set_device(local_rank)
    
    # Detect ZeRO stage and create experiment name
    zero_stage = get_zero_stage_from_config(args.deepspeed_config)
    experiment_name = create_experiment_name(world_size, zero_stage)
    
    # Set output directory based on experiment
    if args.output_dir is None:
        args.output_dir = f"./checkpoints/{experiment_name}"
    
    # Results directory for metrics
    results_dir = "results"
    if is_main_process:
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Print info only from main process
    if is_main_process:
        print("\n" + "="*70)
        print(f"DEEPSPEED ZeRO-{zero_stage} TRAINING")
        print("="*70)
        print(f"\nExperiment: {experiment_name}")
        print(f"GPUs: {world_size}")
        print(f"ZeRO Stage: {zero_stage}")
        print(f"Config: {args.deepspeed_config}")
        print(f"Output: {args.output_dir}")
        print(f"Metrics File: {results_dir}/{experiment_name}_metrics.csv")
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
    
    # DeepSpeed handles device placement, don't use device_map
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
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
    model.enable_input_require_grads()
    
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
    
    num_samples = len(tokenized)
    if is_main_process:
        print(f"Dataset ready: {num_samples:,} samples")
    
    # Calculate epochs
    calculated_epochs = calculate_epochs(args, num_samples, world_size)
    if args.num_train_epochs is None:
        args.num_train_epochs = calculated_epochs
        if is_main_process:
            print(f"Auto-calculated epochs: {calculated_epochs:.2f} (based on target_total_steps={args.target_total_steps})")
    
    effective_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps * world_size
    steps_per_epoch = num_samples / effective_batch_size
    total_expected_steps = int(args.num_train_epochs * steps_per_epoch)
    if is_main_process:
        print(f"Expected total steps: {total_expected_steps:,} (epochs={args.num_train_epochs:.2f} × {steps_per_epoch:.0f} steps/epoch)")
    
    # Load cumulative metrics if resuming (only on main process)
    resume_checkpoint = None
    cumulative_time_hours = 0.0
    cumulative_global_steps = 0
    
    if is_main_process and args.resume_from_checkpoint:
        if os.path.exists(args.output_dir):
            checkpoints = [d for d in os.listdir(args.output_dir) 
                          if d.startswith("checkpoint-") and os.path.isdir(os.path.join(args.output_dir, d))]
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[1]))
                resume_checkpoint = os.path.join(args.output_dir, latest_checkpoint)
                print(f"\n✓ Found checkpoint: {resume_checkpoint}")
                
                # Load cumulative metrics from experiment-specific CSV
                cumulative_time_hours, cumulative_global_steps = load_cumulative_metrics(
                    args.output_dir, results_dir, experiment_name
                )
                if cumulative_time_hours > 0 or cumulative_global_steps > 0:
                    print(f"  Loaded cumulative: {cumulative_time_hours:.2f} hours, {cumulative_global_steps:,} steps")
                
                print("  Training will resume from this checkpoint")
            else:
                print("\n✓ No checkpoint found, starting fresh training")
    
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
        
        # Memory optimization
        gradient_checkpointing=True,
        fp16=True,
        
        # Logging
        logging_steps=10,
        logging_dir=f"{args.output_dir}/logs",
        
        # Saving
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        
        # Distributed settings
        ddp_find_unused_parameters=False,
        local_rank=local_rank,
        
        # Disable reporting
        report_to="none",
        
        # Add these for better multi-GPU performance:
        dataloader_pin_memory=True,
        dataloader_num_workers=4,  # Important for data loading speed
        ddp_timeout=1800000,  # 30 minutes timeout
        
        # Better logging for analysis:
        logging_steps=5,  # More frequent logging
        eval_steps=200,   # Add evaluation if you have validation set
        evaluation_strategy="steps",  # Monitor performance
        
        # Better saving:
        save_steps=500,
        save_total_limit=2,
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
    
    if is_main_process:
        print("✓ Trainer configured with DeepSpeed ZeRO-2")
        print(f"\nEffective batch size: {effective_batch_size}")
        print(f"  = {args.per_device_train_batch_size} (per_device) × {args.gradient_accumulation_steps} (grad_accum) × {world_size} (GPUs)")
        print(f"Target epochs: {args.num_train_epochs:.2f} ({total_expected_steps:,} total steps)")
        if cumulative_global_steps > 0:
            remaining_steps = total_expected_steps - cumulative_global_steps
            print(f"Remaining steps: {remaining_steps:,} (from step {cumulative_global_steps:,})")
    
    # Train
    if is_main_process:
        print("\n" + "="*70)
        print(f"STARTING TRAINING: {experiment_name.upper()}")
        print("="*70)
        print()
    
    session_start_time = time.time()
    
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    
    session_time = time.time() - session_start_time
    
    # Synchronize all processes before collecting metrics
    if is_distributed:
        torch.distributed.barrier()
    
    # Save and collect metrics (only main process)
    if is_main_process:
        session_global_steps = trainer.state.global_step - cumulative_global_steps
        total_global_steps = trainer.state.global_step
        
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
        
        # Get final loss from training history
        train_history = trainer.state.log_history
        final_loss = None
        for entry in reversed(train_history):
            if 'loss' in entry:
                final_loss = entry['loss']
                break
        
        # Session metrics
        session_time_hours = session_time / 3600
        total_time_hours = cumulative_time_hours + session_time_hours
        session_samples = session_global_steps * effective_batch_size
        total_samples_processed = total_global_steps * effective_batch_size
        
        # Get memory usage
        peak_memory_gb = torch.cuda.max_memory_allocated() / 1e9
        
        # Prepare metrics dictionary for THIS SESSION
        metrics = {
            "experiment": experiment_name,
            "num_gpus": world_size,
            "zero_stage": zero_stage,
            "strategy": f"deepspeed_zero{zero_stage}",
            "training_time_hours": session_time_hours,  # This session only
            "total_steps": total_global_steps,  # Cumulative position
            "samples_processed": session_samples,  # This session only
            "samples_per_second": session_samples / session_time if session_time > 0 else 0,
            "cumulative_time_hours": total_time_hours,  # Total across all sessions
            "cumulative_samples_per_second": total_samples_processed / (total_time_hours * 3600) if total_time_hours > 0 else 0,
            "peak_memory_gb": peak_memory_gb,
            "final_loss": final_loss if final_loss is not None else 0.0,
            "target_epochs": args.num_train_epochs,
            "actual_epochs": total_global_steps / steps_per_epoch,
        }
        
        # Print and save metrics
        print_metrics_summary(metrics)
        save_training_metrics(metrics, results_dir=results_dir)
        
        # Final summary
        print("\n" + "="*70)
        print(f"{experiment_name.upper()} TRAINING COMPLETE!")
        print("="*70)
        print(f"\nSession time: {session_time_hours:.2f} hours")
        print(f"Total cumulative time: {total_time_hours:.2f} hours")
        print(f"Session throughput: {metrics['samples_per_second']:.1f} samples/sec")
        print(f"Cumulative throughput: {metrics['cumulative_samples_per_second']:.1f} samples/sec")
        print(f"Memory: {metrics['peak_memory_gb']:.2f} GB")
        print(f"Steps: {total_global_steps:,} / {total_expected_steps:,}")
        print(f"Epochs: {metrics['actual_epochs']:.2f} / {metrics['target_epochs']:.2f}")
        print(f"\nCheckpoints: {args.output_dir}")
        print(f"Metrics saved to:")
        print(f"  - {results_dir}/{experiment_name}_metrics.csv (experiment-specific)")
        print(f"  - {results_dir}/all_experiments_metrics.csv (combined)")
        print()


if __name__ == "__main__":
    main()
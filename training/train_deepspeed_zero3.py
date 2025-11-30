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
    TrainerCallback
)
import csv
from peft import LoraConfig, get_peft_model, TaskType

from utils import (
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
        default="meta-llama/Llama-2-13b-hf",
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

    parser.add_argument(
        "--quick_test",
        action="store_true",
        help="Run quick test with small subset (for testing pipeline)"
    )

    parser.add_argument(
        "--save_steps",
        type=int,
        default=None,
        help="Save checkpoint every N steps (in addition to epochs)"
    )
    
    return parser.parse_args()

def save_checkpoint_metrics(metrics, file_path="results/checkpoint_metrics.csv"):  
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    file_exists = os.path.isfile(file_path)
    
    with open(file_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)

class SaveMetricsOnCheckpoint(TrainerCallback):
 
    def __init__(self):
        self.start_time = time.time()
        self.first_step = True

    def on_step_begin(self, args, state, control, **kwargs):
        # Run diagnostics on first step only
        if self.first_step and args.local_rank <= 0:
            self.first_step = False
            print("\n" + "="*70)
            print("DEEPSPEED DIAGNOSTIC (After Initialization)")
            print("="*70)

            # Now trainer.model.optimizer should exist
            if hasattr(kwargs.get('model'), 'optimizer'):
                print("✓ DeepSpeed optimizer detected")

            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"  GPU Memory allocated: {allocated:.2f} GB")
            print(f"  GPU Memory reserved: {reserved:.2f} GB")
            print("="*70 + "\n")

        return control
 
    def on_save(self, args, state, control, **kwargs):
        # Time since training started
        elapsed_seconds = time.time() - self.start_time
        self.start_time = time.time()
        
        # Safe extraction of last log entry
        recent_logs = state.log_history[-100:]
        last_log = next((log for log in reversed(recent_logs) if "loss" in log), {})
        
        metrics = {
            "global_step": state.global_step,
            "training_time_hours": elapsed_seconds / 3600,
            "loss": last_log.get("loss"),
            "learning_rate": last_log.get("learning_rate"),
            "epoch": state.epoch,
            "peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0,
        }
        
        save_checkpoint_metrics(metrics)
        torch.cuda.empty_cache()

def main():
    """Main training function."""
    args = parse_args()
    
    # Detect GPU count and ZeRO stage
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    zero_stage = get_zero_stage_from_config(args.deepspeed)
    experiment_name = create_experiment_name(num_gpus, zero_stage)
    
    if args.output_dir is None:
        args.output_dir = f"./checkpoints/{experiment_name}"
    
    # Only print on main process
    if args.local_rank <= 0:
        print("\n" + "="*70)
        print(f"DEEPSPEED ZeRO-{zero_stage} TRAINING")
        print("="*70)
        print(f"\nExperiment: {experiment_name}")
        print(f"GPUs: {num_gpus}")
        print(f"Output: {args.output_dir}\n")
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    
    # Load tokenizer
    if args.local_rank <= 0:
        print("[1/5] Loading tokenizer...")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model - NO device_map with DeepSpeed!
    if args.local_rank <= 0:
        print("[2/5] Loading model...")
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        use_cache=False
    )
    
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
    
    # Enable input gradients for LoRA
    model.enable_input_require_grads()
    
    if args.local_rank <= 0:
        model.print_trainable_parameters()
    
    # Load dataset
    if args.local_rank <= 0:
        print("[4/5] Loading dataset...")
    
    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset not found: {args.dataset_path}")
    
    dataset = load_from_disk(args.dataset_path)
    
    if args.quick_test:
        print("\nQUICK TEST MODE: 100 samples, 1 epoch\n")
        dataset = dataset.select(range(100))
        args.num_train_epochs = 1
    
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
        print("[5/5] Configuring training with DeepSpeed ZeRO-3...")

    deepspeed_config = os.path.abspath(args.deepspeed)
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        # Optimization
        fp16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": True},  # Important for ZeRO-3!
        # Logging
        logging_steps=10,
        logging_dir=f"{args.output_dir}/logs",
        # Saving
        save_strategy="steps" if args.save_steps else "epoch",
        save_steps=args.save_steps if args.save_steps else 500,
        save_total_limit=2,
        # DeepSpeed
        deepspeed=deepspeed_config,
        # Important for DeepSpeed ZeRO-3 with LoRA
        ddp_find_unused_parameters=False,
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
        callbacks=[SaveMetricsOnCheckpoint()]
    )
    
    if args.local_rank <= 0:
        effective_bs = args.per_device_train_batch_size * args.gradient_accumulation_steps * num_gpus
        print(f"✓ Effective batch size: {effective_bs}\n")
        print("="*70)
        print("STARTING TRAINING")
        print("="*70 + "\n")
    
    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time
    
    # Save model
    if args.local_rank <= 0:
        final_dir = f"{args.output_dir}/final"
        trainer.save_model(final_dir)
        tokenizer.save_pretrained(final_dir)
        
        # Collect and save metrics
        num_samples = len(tokenized)
        train_history = trainer.state.log_history
        final_loss = next((entry['loss'] for entry in reversed(train_history) if 'loss' in entry), None)
        
        metrics = {
            "experiment": experiment_name,
            "num_gpus": num_gpus,
            "zero_stage": zero_stage,
            "strategy": f"deepspeed_zero{zero_stage}",
            "training_time_hours": training_time / 3600,
            "samples_per_second": (num_samples * args.num_train_epochs) / training_time,
            "peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9,
            "final_loss": final_loss or 0.0,
        }
        
        print_metrics_summary(metrics)
        save_training_metrics(metrics)
        
        print(f"\n✓ Training complete: {training_time/3600:.2f} hours")
        print(f"✓ Model saved to {final_dir}\n")


if __name__ == "__main__":
    main()
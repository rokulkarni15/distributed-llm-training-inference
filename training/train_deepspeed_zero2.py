#!/usr/bin/env python
"""
DeepSpeed ZeRO-2 training: Works with 1, 2, or 4 GPUs

Usage:
    # 1 GPU
    deepspeed --num_gpus=1 training/train_zero2.py --deepspeed configs/ds_config_zero2.json
    
    # 4 GPUs
    deepspeed --num_gpus=4 training/train_zero2.py --deepspeed configs/ds_config_zero2.json
"""

import os
import argparse
import time
import torch
import csv
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback
)

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
        description="Fine-tune with DeepSpeed ZeRO-2"
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
        "--max_steps",
        type=int,
        default=-1,
        help="Maximum training steps (overrides num_train_epochs if set)"
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
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    
    return parser.parse_args()


def save_checkpoint_metrics(metrics, file_path=f"results/checkpoint_metrics{parse_args().local_rank}.csv"):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    file_exists = os.path.isfile(file_path)
    
    with open(file_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        
        # Write header if file didn't exist
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(metrics)
        
        
class SaveMetricsOnCheckpoint(TrainerCallback):

    def __init__(self):
        self.start_time = time.time()

    def on_save(self, args, state, control, **kwargs):
        # Time since training started
        elapsed_seconds = time.time() - self.start_time
        
        # Safe extraction of last log entry
        last_log = state.log_history[-1] if len(state.log_history) > 0 else {}
        
        metrics = {
            "global_step": state.global_step,
            "training_time_hours": elapsed_seconds / 3600,
            "loss": last_log.get("loss"),
            "learning_rate": last_log.get("learning_rate"),
            "epoch": state.epoch,
            "peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9,
        }
        
        save_checkpoint_metrics(metrics)
        
def main():
    """Main training function."""
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
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
            print("NOTE: Running on 1 GPU with ZeRO-2 optimizer state partitioning")
        else:
            print(f"NOTE: Running on {num_gpus} GPUs with ZeRO-2 optimizer state partitioning across all GPUs")
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
    torch.cuda.empty_cache()
    
    if args.local_rank <= 0:
        print("Model loaded")
    
    if args.local_rank <= 0:
        print(f"[3/5] Applying LoRA (r={args.lora_r})...")

   
    lora_config = LoraConfig(
        r=args.lora_r,                           # From command line (32)
        lora_alpha=args.lora_r * 2,              # HARD-CODED: 2x rank = 64
        lora_dropout=0.05,                       # HARD-CODED: slight dropout
        
        # HARD-CODED: Target ALL attention + MLP modules
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",      # Attention
            "gate_proj", "up_proj", "down_proj"          # MLP layers
        ],
        
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)

    # Enable gradient checkpointing for LoRA
    model.enable_input_require_grads()

    if args.local_rank <= 0:
        model.print_trainable_parameters()
        print("LoRA applied")
    
    # Load dataset
    if args.local_rank <= 0:
        print("[4/5] Loading dataset...")

    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset not found: {args.dataset_path}")

    dataset = load_from_disk(args.dataset_path)

    if isinstance(dataset, dict) or hasattr(dataset, 'keys'):
        if "validation" not in dataset:
            print("Splitting train set into train/validation...")
            split_dataset = dataset["train"].train_test_split(test_size=0.05, seed=42)
            train_dataset = split_dataset["train"]
            eval_dataset = split_dataset["test"]
        else:
            train_dataset = dataset["train"]
            eval_dataset = dataset["validation"]
    else:
        print("Splitting dataset into train/validation...")
        split_dataset = dataset.train_test_split(test_size=0.05, seed=42)
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]

    print(f"Train samples: {len(train_dataset):,}")
    print(f"Validation samples: {len(eval_dataset):,}")

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding=False,  # Don't pad during tokenization, let the data collator handle padding
        )

    # Tokenize the train and evaluation datasets separately
    if args.local_rank <= 0:
        print("Tokenizing train dataset...")
    train_dataset = train_dataset.map(
        tokenize,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train dataset"
    )

    if args.local_rank <= 0:
        print("Tokenizing evaluation dataset...")
    eval_dataset = eval_dataset.map(
        tokenize,
        batched=True,
        remove_columns=eval_dataset.column_names,
        desc="Tokenizing evaluation dataset"
    )

    if args.local_rank <= 0:
        print(f"Tokenized datasets ready:")
        print(f"Train samples: {len(train_dataset):,}")
        print(f"Validation samples: {len(eval_dataset):,}")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        
        learning_rate=args.learning_rate,
        lr_scheduler_type="constant",
        warmup_ratio=0.01,  # Reduced from 0.05 for faster warmup
        
        optim="adamw_torch",
        weight_decay=0.01,
        max_grad_norm=1.0,
        
        fp16=True,
        gradient_checkpointing=True,
        
        # Critical changes for stability
        eval_strategy="steps",
        eval_steps=500,  # More frequent evaluation to monitor progress
        per_device_eval_batch_size=4,
        
        logging_steps=25,  # More frequent logging to monitor progress
        logging_first_step=True,
        
        save_strategy="steps",
        save_steps=500,  # More frequent saves
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        
        # Key changes to prevent hanging
        dataloader_num_workers=0,  # Disable multiprocessing to avoid tokenizer fork issues
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        
        # Additional stability settings
        dataloader_drop_last=True,  # Avoid partial batches that can cause issues
        disable_tqdm=False,  # Keep progress bars
        
        report_to="none",
        local_rank=args.local_rank,
        seed=42,
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
   
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,      # ADDED THIS for evaluation
        data_collator=data_collator,
        callbacks=[SaveMetricsOnCheckpoint()]
    )
    
    if args.local_rank <= 0:
        print("✓ Trainer configured")
        effective_bs = args.per_device_train_batch_size * args.gradient_accumulation_steps * num_gpus
        print(f"Effective batch size: {effective_bs}")
    
    # Train
    # Train
    if args.local_rank <= 0:
        print("\n" + "="*70)
        print(f"STARTING TRAINING: {experiment_name.upper()}")
        if args.resume_from_checkpoint:
            print(f"RESUMING FROM: {args.resume_from_checkpoint}")
        print("="*70)
        print()

    start_time = time.time()

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

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
        
        num_samples = len(train_dataset)
        
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
        save_training_metrics(metrics, csv_path=f"results/training_metrics_gpu{args.local_rank}.csv")
        
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
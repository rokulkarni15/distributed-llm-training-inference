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
        default="./checkpoints/deepspeed_zero2",
        help="Directory to save checkpoints"
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
    
    # Only print from main process
    if is_main_process:
        print("\n" + "="*70)
        print(f"DEEPSPEED ZeRO-2 TRAINING: PYTORCH + LORA + DeepSpeed ({world_size} GPU{'s' if world_size > 1 else ''})")
        print("="*70)
        print("\nNOTE: Using DeepSpeed ZeRO-2 for optimizer + gradient state partitioning")
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
    
    # Adjust output directory based on number of GPUs
    output_dir = f"{args.output_dir}_{world_size}gpu" if world_size > 1 else f"{args.output_dir}_1gpu"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
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
        logging_dir=f"{output_dir}/logs",
        
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
        if os.path.exists(output_dir):
            checkpoints = [d for d in os.listdir(output_dir) 
                          if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))]
            if checkpoints:
                # Get the latest checkpoint
                latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[1]))
                resume_checkpoint = os.path.join(output_dir, latest_checkpoint)
                print(f"\n✓ Found checkpoint: {resume_checkpoint}")
                print("  Training will resume from this checkpoint")
            else:
                print("\n✓ No checkpoint found, starting fresh training")
    
    if is_main_process:
        print("Trainer ready with DeepSpeed ZeRO-2")
        total_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps * world_size
        print(f"\nEffective batch size: {total_batch_size}")
        print(f"  = {args.per_device_train_batch_size} (per_device) × {args.gradient_accumulation_steps} (grad_accum) × {world_size} (GPUs)")
    
    # Train
    if is_main_process:
        print("\n" + "="*70)
        print("STARTING DEEPSPEED ZeRO-2 TRAINING")
        print("="*70)
    
    import time
    start_time = time.time()
    
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    
    training_time = time.time() - start_time
    
    # Save (only main process saves)
    if is_main_process:
        print("\n" + "="*70)
        print("SAVING MODEL")
        print("="*70)
        
        final_dir = f"{output_dir}/final"
        trainer.save_model(final_dir)
        tokenizer.save_pretrained(final_dir)
        
        print(f"\nTraining complete!")
        print(f"Time: {training_time/3600:.2f} hours")
        print(f"Saved to: {final_dir}")


if __name__ == "__main__":
    main()
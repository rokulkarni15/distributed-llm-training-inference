#!/usr/bin/env python3
"""
DeepSpeed ZeRO-1 training: PyTorch + LoRA + DeepSpeed ZeRO-1 on single GPU.

Usage:
    old command
    # python training/train_deepspeed_zero1.py --dataset_path ./data/glaive_code_full
    new command to make sure the model resumes previous checkpoints
    python training/train_deepspeed_zero1.py --dataset_path ./data/glaive_code_full --resume_from_checkpoint
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
        description="Fine-tune Llama 2 7B with LoRA + DeepSpeed ZeRO-1 (1 GPU)"
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
        default="./checkpoints/deepspeed_zero1_1gpu",
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
        default="./training/ds_config_zero1.json",
        help="Path to DeepSpeed config file"
    )
    
    parser.add_argument(
        "--resume_from_checkpoint",
        action="store_true",
        help="Resume from last checkpoint if available"
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    print("\n" + "="*70)
    print("DEEPSPEED ZeRO-1 TRAINING: PYTORCH + LORA + DeepSpeed (1 GPU)")
    print("="*70)
    print("\nNOTE: Using DeepSpeed ZeRO-1 for optimizer state partitioning")
    print("Expected: Lower memory usage than baseline, similar speed on 1 GPU")
    print()
    
    # Check GPU
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load tokenizer
    print("\n[1/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    print("\n[2/5] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    # Apply LoRA
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
    model.print_trainable_parameters()
    
    # Load dataset
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
        desc="Tokenizing"
    )
    
    print(f"Dataset ready: {len(tokenized):,} samples")
    
    # Training setup with DeepSpeed
    print("\n[5/5] Configuring training with DeepSpeed...")
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        
        # DeepSpeed integration - ONLY CHANGE NEEDED
        deepspeed=args.deepspeed_config,
        
        # Memory optimization (DeepSpeed will handle fp16)
        gradient_checkpointing=True,
        
        # Logging
        logging_steps=10,
        logging_dir=f"{args.output_dir}/logs",
        
        # Saving
        save_strategy="epoch",
        save_total_limit=2,
        
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
    
    print("Trainer ready with DeepSpeed ZeRO-1")
    print(f"\nEffective batch size: {args.per_device_train_batch_size * args.gradient_accumulation_steps}")
    
    # Train
    print("\n" + "="*70)
    print("STARTING DEEPSPEED ZeRO-1 TRAINING")
    print("="*70)
    
    import time
    start_time = time.time()
    
    trainer.train()
    
    training_time = time.time() - start_time
    
    # Save
    print("\n" + "="*70)
    print("SAVING MODEL")
    print("="*70)
    
    final_dir = f"{args.output_dir}/final"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    
    print(f"\nTraining complete!")
    print(f"Time: {training_time/3600:.2f} hours")
    print(f"Saved to: {final_dir}")


if __name__ == "__main__":
    main()
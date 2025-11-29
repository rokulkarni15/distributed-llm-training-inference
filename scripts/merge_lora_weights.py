#!/usr/bin/env python
"""
Merge LoRA adapter weights back into base model.

Creates complete fine-tuned model that vLLM can load.

Usage:
    python scripts/merge_lora_weights.py \
        --lora_weights ./checkpoints/baseline_1gpu/final \
        --output_dir ./merged_model/baseline
"""

import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge_lora_weights(base_model_name, lora_weights_path, output_dir):
    """
    Merge LoRA adapters into base model.
    
    Args:
        base_model_name: HuggingFace model name (e.g., meta-llama/Llama-2-7b-hf)
        lora_weights_path: Path to LoRA checkpoint directory
        output_dir: Where to save merged model
    """
    print("\n" + "="*70)
    print("MERGING LORA WEIGHTS INTO BASE MODEL")
    print("="*70)
    print(f"\nBase model: {base_model_name}")
    print(f"LoRA weights: {lora_weights_path}")
    print(f"Output: {output_dir}")
    print()
    
    # Verify LoRA checkpoint exists
    if not os.path.exists(lora_weights_path):
        raise FileNotFoundError(f"LoRA checkpoint not found: {lora_weights_path}")
    
    # Load base model
    print("[1/4] Loading base model...")
    print("This may take a few minutes...")
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    print("✓ Base model loaded")
    
    # Load LoRA adapters
    print("\n[2/4] Loading LoRA adapters...")
    
    model = PeftModel.from_pretrained(
        base_model,
        lora_weights_path,
        torch_dtype=torch.float16,
    )
    print("✓ LoRA adapters loaded")
    
    # Merge
    print("\n[3/4] Merging LoRA weights into base model...")
    print("This combines the adapters with base weights...")
    
    merged_model = model.merge_and_unload()
    print("✓ Merge complete")
    
    # Save merged model
    print(f"\n[4/4] Saving merged model...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    merged_model.save_pretrained(
        output_dir,
        safe_serialization=True,  # Use safetensors format
    )
    print(f"✓ Model saved to {output_dir}")
    
    # Save tokenizer
    print("\nSaving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.save_pretrained(output_dir)
    print("✓ Tokenizer saved")
    
    # Print summary
    print("\n" + "="*70)
    print("MERGE COMPLETE!")
    print("="*70)
    print(f"\nMerged model location: {output_dir}")
    print(f"Model size: ~14GB")
    print("\nThis model can now be used with vLLM:")
    print(f"  python -m vllm.entrypoints.openai.api_server --model {output_dir}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA weights into base model for inference"
    )
    
    parser.add_argument(
        "--base_model",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="Base model name from HuggingFace"
    )
    
    parser.add_argument(
        "--lora_weights",
        type=str,
        required=True,
        help="Path to LoRA checkpoint directory (e.g., ./checkpoints/baseline_1gpu/final)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for merged model (e.g., ./merged_model/baseline)"
    )
    
    args = parser.parse_args()
    
    merge_lora_weights(
        args.base_model,
        args.lora_weights,
        args.output_dir
    )


if __name__ == "__main__":
    main()
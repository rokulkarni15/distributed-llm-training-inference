#!/usr/bin/env python3
"""
Prepare Glaive-Code-Assistant dataset for training.

This script:
1. Downloads the Glaive-Code-Assistant dataset from HuggingFace
2. Formats multi-turn conversations for Llama 2 chat format
3. Saves preprocessed dataset for fast loading during training
4. Verifies dataset size and integrity

Usage:
    # Full dataset (~136K samples)
    python scripts/prepare_dataset.py
    
    # Subset for testing
    python scripts/prepare_dataset.py --num_samples 10000
"""

import os
import argparse
from datasets import load_dataset
from pathlib import Path


def format_conversation_for_llama2(example):
    """
    Format Glaive conversation into Llama 2 chat template.
    
    Llama 2 format:
    <s>[INST] <<SYS>>
    {system_prompt}
    <</SYS>>
    
    {user_message} [/INST] {assistant_response}</s>
    <s>[INST] {user_message} [/INST] {assistant_response}</s>
    
    Args:
        example: Dataset example with 'system' and 'chat' fields
    
    Returns:
        Dict with 'text' field containing formatted conversation
    """
    system = example.get('system', 'You are a helpful coding assistant.')
    chat = example['chat']
    
    # Start conversation with system prompt
    conversation = f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n"
    
    # Add conversation turns
    for i, turn in enumerate(chat):
        role = turn['role']
        content = turn['content']
        
        if role == 'user':
            if i == 0:
                # First user message (already has system prompt)
                conversation += f"{content} [/INST] "
            else:
                # Subsequent user messages start new turn
                conversation += f"<s>[INST] {content} [/INST] "
        else:  # assistant
            conversation += f"{content}</s>"
    
    return {"text": conversation}


def prepare_glaive_dataset(num_samples=None, output_dir="./data"):
    """
    Download and prepare Glaive-Code-Assistant dataset.
    
    Args:
        num_samples: Number of samples to use (None = full dataset)
        output_dir: Directory to save preprocessed dataset
    
    Returns:
        Prepared dataset
    """
    print("=" * 70)
    print("PREPARING GLAIVE-CODE-ASSISTANT DATASET")
    print("=" * 70)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print("\n[1/4] Loading dataset from HuggingFace...")
    print("Dataset: glaiveai/glaive-code-assistant")
    
    try:
        dataset = load_dataset("glaiveai/glaive-code-assistant", split="train")
        print(f"✓ Loaded {len(dataset):,} samples")
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        print("\nTroubleshooting:")
        print("1. Check internet connection on cluster")
        print("2. Verify HuggingFace datasets library is installed")
        print("3. Try: huggingface-cli login (if dataset requires auth)")
        return None
    
    # Subsample if requested
    if num_samples:
        print(f"\n[2/4] Subsampling to {num_samples:,} samples...")
        dataset = dataset.select(range(min(num_samples, len(dataset))))
        print(f"✓ Using {len(dataset):,} samples")
    else:
        print(f"\n[2/4] Using full dataset ({len(dataset):,} samples)")
    
    # Format conversations
    print("\n[3/4] Formatting conversations for Llama 2...")
    print("Converting multi-turn conversations to Llama 2 chat format...")
    
    original_columns = dataset.column_names
    dataset = dataset.map(
        format_conversation_for_llama2,
        remove_columns=original_columns,
        desc="Formatting conversations",
        num_proc=4  # Use 4 processes for speed
    )
    
    print(f"✓ Formatted {len(dataset):,} conversations")
    
    # Show example
    print("\n" + "-" * 70)
    print("EXAMPLE FORMATTED CONVERSATION (first 500 chars):")
    print("-" * 70)
    example_text = dataset[0]['text']
    print(example_text[:500] + "..." if len(example_text) > 500 else example_text)
    print("-" * 70)
    
    # Save preprocessed dataset
    sample_suffix = f"_{num_samples//1000}k" if num_samples else "_full"
    output_path = os.path.join(output_dir, f"glaive_code{sample_suffix}")
    
    print(f"\n[4/4] Saving preprocessed dataset...")
    print(f"Output path: {output_path}")
    
    dataset.save_to_disk(output_path)
    
    # Calculate size
    total_size = sum(
        os.path.getsize(os.path.join(dirpath, filename))
        for dirpath, dirnames, filenames in os.walk(output_path)
        for filename in filenames
    )
    size_mb = total_size / (1024 * 1024)
    
    print(f"✓ Saved successfully")
    print(f"✓ Dataset size: {size_mb:.1f} MB")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Samples:        {len(dataset):,}")
    print(f"Size:           {size_mb:.1f} MB")
    print(f"Location:       {output_path}")
    print(f"Format:         Llama 2 chat format")
    print("=" * 70)
    
    print("\n✓ Dataset preparation complete!")
    print(f"\nTo use in training, load from: {output_path}")
    print("Example:")
    print(f"    from datasets import load_from_disk")
    print(f"    dataset = load_from_disk('{output_path}')")
    
    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Glaive-Code-Assistant dataset for training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Prepare full dataset (~136K samples)
    python scripts/prepare_dataset.py
    
    # Prepare subset for quick testing
    python scripts/prepare_dataset.py --num_samples 10000
    
    # Specify custom output directory
    python scripts/prepare_dataset.py --output_dir ./my_data
        """
    )
    
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to use (default: full dataset ~136K)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data",
        help="Output directory for preprocessed dataset (default: ./data)"
    )
    
    args = parser.parse_args()
    
    # Prepare dataset
    dataset = prepare_glaive_dataset(
        num_samples=args.num_samples,
        output_dir=args.output_dir
    )
    
    if dataset:
        print("\n✓ SUCCESS! Dataset is ready for training.")
    else:
        print("\n✗ FAILED! Check errors above.")
        exit(1)


if __name__ == "__main__":
    main()
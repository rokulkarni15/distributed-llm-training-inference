#!/usr/bin/env python
"""
Start vLLM server for inference.

Wrapper around vLLM's OpenAI-compatible API server.

Usage:
    # Single GPU
    python inference/serve_vllm.py --model ./merged_model/zero3_4gpu
    
    # Multi-GPU with tensor parallelism
    python inference/serve_vllm.py \
        --model ./merged_model/zero3_4gpu \
        --tensor_parallel_size 4
"""

import argparse
import os
import sys
import subprocess


def start_vllm_server(
    model_path,
    tensor_parallel_size=1,
    host="0.0.0.0",
    port=8000,
    max_model_len=2048,
    gpu_memory_utilization=0.9,
):
    """
    Start vLLM server with OpenAI-compatible API.
    
    Args:
        model_path: Path to merged model directory
        tensor_parallel_size: Number of GPUs for tensor parallelism
        host: Server host
        port: Server port
        max_model_len: Maximum sequence length
        gpu_memory_utilization: GPU memory utilization (0.0-1.0)
    """
    print("="*70)
    print("STARTING vLLM SERVER")
    print("="*70)
    print(f"\nModel: {model_path}")
    print(f"Tensor Parallel Size: {tensor_parallel_size} GPU(s)")
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"Max sequence length: {max_model_len}")
    print(f"GPU memory utilization: {gpu_memory_utilization}")
    print()
    
    # Verify model exists
    if not os.path.exists(model_path):
        print(f"✗ Error: Model not found at {model_path}")
        print("\nDid you merge LoRA weights?")
        print("Run: python scripts/merge_lora_weights.py \\")
        print("       --lora_weights ./checkpoints/YOUR_CHECKPOINT/final \\")
        print("       --output_dir ./merged_model/YOUR_MODEL")
        sys.exit(1)
    
    # Verify vLLM installed
    try:
        import vllm
        print(f"vLLM version: {vllm.__version__}")
    except ImportError:
        print("Error: vLLM not installed")
        print("Install with: pip install vllm")
        sys.exit(1)
    
    # Check GPU availability
    import torch
    if not torch.cuda.is_available():
        print("Warning: CUDA not available!")
        print("vLLM requires GPU to run efficiently.")
    else:
        num_gpus = torch.cuda.device_count()
        print(f"GPUs available: {num_gpus}")
        
        if tensor_parallel_size > num_gpus:
            print(f"\nError: Requested {tensor_parallel_size} GPUs but only {num_gpus} available")
            sys.exit(1)
        
        for i in range(num_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    print("\n" + "="*70)
    print("LAUNCHING SERVER")
    print("="*70)
    print("\nServer will be available at:")
    print(f"  http://{host}:{port}")
    print(f"  API endpoint: http://{host}:{port}/v1/completions")
    print("\nPress Ctrl+C to stop the server")
    print()
    
    # Build vLLM command
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--host", host,
        "--port", str(port),
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--dtype", "float16",
    ]
    
    # Run server
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n\nServer stopped by user.")
    except Exception as e:
        print(f"\nError starting server: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Start vLLM inference server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single GPU
    python inference/serve_vllm.py --model ./merged_model/zero3_4gpu
    
    # 2 GPUs with tensor parallelism
    python inference/serve_vllm.py \
        --model ./merged_model/zero3_4gpu \
        --tensor_parallel_size 2
    
    # 4 GPUs with tensor parallelism
    python inference/serve_vllm.py \
        --model ./merged_model/zero3_4gpu \
        --tensor_parallel_size 4 \
        --port 8000
        """
    )
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to merged model directory"
    )
    
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism (default: 1)"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host (default: 0.0.0.0)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port (default: 8000)"
    )
    
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=2048,
        help="Maximum sequence length (default: 2048)"
    )
    
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization fraction (default: 0.9)"
    )
    
    args = parser.parse_args()
    
    start_vllm_server(
        model_path=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        host=args.host,
        port=args.port,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )


if __name__ == "__main__":
    main()
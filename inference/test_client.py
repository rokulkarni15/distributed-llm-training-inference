#!/usr/bin/env python
"""
Simple test client for vLLM server.

Tests that the server is responding correctly to requests.

Usage:
    # Start vLLM server first, then:
    python inference/test_client.py --base_url http://localhost:8000
"""

import argparse
import sys

try:
    from openai import OpenAI
except ImportError:
    print("Error: openai package not installed")
    print("Install with: pip install openai")
    sys.exit(1)


def test_server(base_url, model_name="fine-tuned-model"):
    """
    Send test requests to vLLM server.
    
    Args:
        base_url: vLLM server URL (e.g., http://localhost:8000)
        model_name: Model identifier for API
    """
    print("="*70)
    print("TESTING vLLM SERVER")
    print("="*70)
    print(f"\nServer URL: {base_url}")
    print(f"Model: {model_name}")
    print()
    
    # Create client
    client = OpenAI(
        base_url=f"{base_url}/v1",
        api_key="dummy",
    )
    
    # Test prompts
    test_prompts = [
        "Write a Python function to check if a number is prime.",
        "Explain how binary search works.",
        "Create a function to reverse a linked list in Python.",
    ]
    
    print("Sending test requests...\n")
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"[Test {i}/{len(test_prompts)}]")
        print(f"Prompt: {prompt}")
        print("-" * 70)
        
        try:
            response = client.completions.create(
                model=model_name,
                prompt=prompt,
                max_tokens=200,
                temperature=0.7,
            )
            
            generated_text = response.choices[0].text
            tokens_generated = response.usage.completion_tokens
            
            print(f"Response: {generated_text[:300]}...")
            print(f"Tokens: {tokens_generated}")
            print("✓ Success")
            
        except Exception as e:
            print(f"✗ Error: {e}")
            return False
        
        print()
    
    print("="*70)
    print("✓ ALL TESTS PASSED")
    print("="*70)
    print("\nvLLM server is working correctly!")
    print()
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Test vLLM server with sample requests"
    )
    
    parser.add_argument(
        "--base_url",
        type=str,
        default="http://localhost:8000",
        help="vLLM server URL"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="fine-tuned-model",
        help="Model name for API"
    )
    
    args = parser.parse_args()
    
    success = test_server(args.base_url, args.model)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
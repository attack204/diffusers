#!/usr/bin/env python3
"""
Script to load and inspect saved prompt embeddings.

Usage:
    python load_prompt_embeddings.py <path_to_pt_file>
    
Example:
    python load_prompt_embeddings.py ./prompt_embeddings/prompt_embeds_20260101_170000.pt
"""

import torch
import json
import sys
import os
from pathlib import Path


def load_embeddings(pt_file_path):
    """
    Load prompt embeddings from a .pt file.
    
    Args:
        pt_file_path: Path to the .pt file containing saved embeddings
        
    Returns:
        dict: Dictionary containing embeddings and metadata
    """
    if not os.path.exists(pt_file_path):
        raise FileNotFoundError(f"File not found: {pt_file_path}")
    
    print(f"Loading embeddings from: {pt_file_path}")
    data = torch.load(pt_file_path, map_location="cpu")
    
    return data


def print_embedding_info(data):
    """Print detailed information about the loaded embeddings."""
    print("\n" + "="*80)
    print("📊 EMBEDDING INFORMATION")
    print("="*80)
    
    # Basic info
    print(f"\n🕐 Timestamp: {data.get('timestamp', 'N/A')}")
    print(f"📝 Prompts: {data.get('prompts', 'N/A')}")
    print(f"🔢 Data type: {data.get('dtype', 'N/A')}")
    
    # Embedding shape and statistics
    prompt_embeds = data["prompt_embeds"]
    print(f"\n📐 Embedding shape: {prompt_embeds.shape}")
    print(f"   - Batch size: {prompt_embeds.shape[0]}")
    print(f"   - Sequence length: {prompt_embeds.shape[1]}")
    print(f"   - Hidden dimension: {prompt_embeds.shape[2]}")
    
    # Statistics
    print(f"\n📈 Statistics:")
    print(f"   - Mean: {prompt_embeds.mean().item():.6f}")
    print(f"   - Std: {prompt_embeds.std().item():.6f}")
    print(f"   - Min: {prompt_embeds.min().item():.6f}")
    print(f"   - Max: {prompt_embeds.max().item():.6f}")
    
    # Attention mask info
    if "encoder_attention_mask" in data:
        attn_mask = data["encoder_attention_mask"]
        print(f"\n🎭 Attention mask shape: {attn_mask.shape}")
        print(f"   - Non-zero elements: {attn_mask.sum().item()}")
        print(f"   - Zero elements: {(attn_mask == 0).sum().item()}")
    
    print("\n" + "="*80 + "\n")


def compare_embeddings(pt_file1, pt_file2):
    """
    Compare two embedding files.
    
    Args:
        pt_file1: Path to first .pt file
        pt_file2: Path to second .pt file
    """
    data1 = load_embeddings(pt_file1)
    data2 = load_embeddings(pt_file2)
    
    embeds1 = data1["prompt_embeds"]
    embeds2 = data2["prompt_embeds"]
    
    print("\n" + "="*80)
    print("🔍 EMBEDDING COMPARISON")
    print("="*80)
    
    print(f"\nFile 1: {pt_file1}")
    print(f"File 2: {pt_file2}")
    
    # Shape comparison
    print(f"\nShape comparison:")
    print(f"  File 1: {embeds1.shape}")
    print(f"  File 2: {embeds2.shape}")
    
    if embeds1.shape == embeds2.shape:
        # Calculate differences
        diff = (embeds1 - embeds2).abs()
        print(f"\n📊 Difference statistics:")
        print(f"   - Mean absolute difference: {diff.mean().item():.6f}")
        print(f"   - Max absolute difference: {diff.max().item():.6f}")
        print(f"   - Cosine similarity: {torch.nn.functional.cosine_similarity(embeds1.flatten(), embeds2.flatten(), dim=0).item():.6f}")
    else:
        print("\n⚠️  Shapes don't match, cannot compute differences")
    
    print("\n" + "="*80 + "\n")


def export_to_numpy(pt_file_path, output_path=None):
    """
    Export embeddings to numpy format.
    
    Args:
        pt_file_path: Path to the .pt file
        output_path: Optional output path for .npy file
    """
    import numpy as np
    
    data = load_embeddings(pt_file_path)
    prompt_embeds = data["prompt_embeds"].numpy()
    
    if output_path is None:
        output_path = pt_file_path.replace(".pt", ".npy")
    
    np.save(output_path, prompt_embeds)
    print(f"✅ Exported to numpy: {output_path}")
    
    # Also save metadata
    metadata_path = output_path.replace(".npy", "_metadata.json")
    metadata = {
        "prompts": data.get("prompts", []),
        "shape": list(prompt_embeds.shape),
        "dtype": str(data.get("dtype", "unknown")),
        "timestamp": data.get("timestamp", "unknown"),
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"📄 Metadata saved to: {metadata_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python load_prompt_embeddings.py <path_to_pt_file> [options]")
        print("\nOptions:")
        print("  --compare <file2>    Compare with another embedding file")
        print("  --export-numpy       Export to numpy format")
        print("\nExample:")
        print("  python load_prompt_embeddings.py ./prompt_embeddings/prompt_embeds_20260101_170000.pt")
        print("  python load_prompt_embeddings.py file1.pt --compare file2.pt")
        print("  python load_prompt_embeddings.py file.pt --export-numpy")
        sys.exit(1)
    
    pt_file = sys.argv[1]
    
    # Handle different modes
    if "--compare" in sys.argv:
        idx = sys.argv.index("--compare")
        if idx + 1 < len(sys.argv):
            pt_file2 = sys.argv[idx + 1]
            compare_embeddings(pt_file, pt_file2)
        else:
            print("Error: --compare requires a second file path")
            sys.exit(1)
    elif "--export-numpy" in sys.argv:
        export_to_numpy(pt_file)
    else:
        # Default: just load and print info
        data = load_embeddings(pt_file)
        print_embedding_info(data)
        
        # Show how to use the embeddings
        print("💡 Usage example:")
        print("```python")
        print("import torch")
        print(f"data = torch.load('{pt_file}')")
        print("prompt_embeds = data['prompt_embeds']")
        print("encoder_attention_mask = data['encoder_attention_mask']")
        print("prompts = data['prompts']")
        print("```\n")


if __name__ == "__main__":
    main()

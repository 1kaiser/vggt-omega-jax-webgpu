# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import math
import torch
import numpy as np
import flax.serialization
import zstandard as zstd

def map_key(pt_key: str):
    parts = pt_key.split('.')
    jax_parts = []
    i = 0
    while i < len(parts):
        part = parts[i]
        
        # Norm parameter renaming: weight -> scale for norm
        if part == "weight" and i == len(parts) - 1:
            prev = parts[i-1] if i > 0 else ""
            is_norm = "norm" in prev or prev == "proj_ln"
            if i >= 2 and parts[i-2] == "embedding_projector" and parts[i-1] == "2":
                is_norm = True
            if is_norm:
                jax_parts.append("scale")
            else:
                jax_parts.append("kernel")
        elif part == "bias" and i == len(parts) - 1:
            jax_parts.append("bias")
        # List of blocks or modules conversion:
        elif part in ("blocks", "frame_blocks", "inter_frame_blocks", "readout_blocks", "trunk", "projects", "resize_layers"):
            next_part = parts[i+1]
            if next_part.isdigit():
                jax_parts.append(f"{part}_{next_part}")
                i += 1
            else:
                jax_parts.append(part)
        elif part == "scratch":
            next_part = parts[i+1]
            if next_part.startswith("layer") or next_part.startswith("refinenet"):
                jax_parts.append(f"scratch_{next_part}")
                i += 1
            else:
                jax_parts.append(part)
        elif part in ("camera_branch", "embedding_projector"):
            next_part = parts[i+1]
            if next_part.isdigit():
                jax_parts.append(f"{part}_{next_part}")
                i += 1
            else:
                jax_parts.append(part)
        else:
            jax_parts.append(part)
        i += 1
        
    return tuple(jax_parts)

def convert_tensor(pt_tensor, jax_path, dtype_name='float32'):
    if isinstance(pt_tensor, torch.Tensor):
        val = pt_tensor.detach().cpu().float().numpy()
    else:
        val = pt_tensor
        
    # Check if this is a weight (kernel) that needs transposition
    if jax_path[-1] == 'kernel':
        if len(val.shape) == 2:
            # Linear layer weight: PT [O, I] -> JAX [I, O]
            val = val.T
        elif len(val.shape) == 4:
            # ConvTranspose or Conv2d
            is_conv_transpose = any(x in jax_path for x in ('resize_layers_0', 'resize_layers_1'))
            if is_conv_transpose:
                # ConvTranspose2d weight: PT [I, O, H, W] -> JAX [H, W, I, O] with spatial flipping
                val = val[:, :, ::-1, ::-1].transpose(2, 3, 0, 1)
            else:
                # Conv2d weight: PT [O, I, H, W] -> JAX [H, W, I, O]
                val = val.transpose(2, 3, 1, 0)
                
    if dtype_name == 'float16':
        val = val.astype(np.float16)
    elif dtype_name == 'bfloat16':
        import ml_dtypes
        val = val.astype(ml_dtypes.bfloat16)
        
    return val

def insert_at_path(d, path, value):
    curr = d
    for part in path[:-1]:
        if part not in curr:
            curr[part] = {}
        curr = curr[part]
    curr[path[-1]] = value

def convert_weights(pt_path: str, output_path: str, dtype_name: str = 'float32'):
    print(f"Loading PyTorch checkpoint from {pt_path}...")
    state_dict = torch.load(pt_path, map_location="cpu")
    if "model" in state_dict:
        state_dict = state_dict["model"]
    
    jax_params = {}
    for pt_key, pt_tensor in state_dict.items():
        # Skip training or non-weight parameters (e.g. resnet buffers if any)
        if "_resnet_" in pt_key or "bias_mask" in pt_key:
            continue
            
        # Apply bias_mask if present in the checkpoint
        mask_key = pt_key + "_mask"
        if mask_key in state_dict:
            pt_tensor = pt_tensor * state_dict[mask_key]
            
        jax_path = map_key(pt_key)
        val = convert_tensor(pt_tensor, jax_path, dtype_name)
        insert_at_path(jax_params, jax_path, val)
        
    # Wrap in "params" structure expected by Flax
    wrapped_params = {"params": jax_params}
    
    print(f"Serializing Flax parameters (dtype: {dtype_name})...")
    serialized_bytes = flax.serialization.to_bytes(wrapped_params)
    
    print(f"Compressing parameters with zstd (original size: {len(serialized_bytes) / (1024*1024):.2f} MB)...")
    cctx = zstd.ZstdCompressor(level=3)
    compressed_bytes = cctx.compress(serialized_bytes)
    
    print(f"Saving to {output_path} (compressed size: {len(compressed_bytes) / (1024*1024):.2f} MB)...")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(compressed_bytes)
        
    print("Weight conversion complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PyTorch VGGT-Omega weights to JAX/Flax msgpack format.")
    parser.add_argument("--pt-path", type=str, required=True, help="Path to PyTorch .pt checkpoint file.")
    parser.add_argument("--output-path", type=str, required=True, help="Path to output JAX .msgpack.zst file.")
    parser.add_argument("--dtype", type=str, choices=["float32", "float16", "bfloat16"], default="float32", help="Target weight data type.")
    args = parser.parse_args()
    
    convert_weights(args.pt_path, args.output_path, args.dtype)

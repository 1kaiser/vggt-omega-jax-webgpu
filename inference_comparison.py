# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # VGGT-Omega PyTorch vs JAX Inference Comparison
# This script/notebook runs both PyTorch and JAX versions of the VGGT-Omega 1B 512 model on real images, compares their predictions, checks mathematical parity, and compares execution times.

# %%
import os
import glob
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

# Force JAX to use CPU backend (since host GPU driver mismatch prevents JAX CUDA init)
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp

from vggt_omega.models import VGGTOmega as PTModel
from vggt_omega.jax.models import VGGTOmega as JAXModel
from vggt_omega.jax.load_weights import load_checkpoint
from vggt_omega.utils.load_fn import load_and_preprocess_images

# %% [markdown]
# ## 0. Download Dataset and Models (if not present)

# %%
import urllib.request
import zipfile

def download_file(url, filepath):
    if os.path.exists(filepath):
        print(f"File {filepath} already exists. Skipping download.")
        return
    print(f"Downloading {url} to {filepath}...")
    try:
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req) as response, open(filepath, 'wb') as out_file:
            chunk_size = 16 * 1024 * 1024  # 16 MB chunks
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
        print(f"Successfully downloaded {filepath}.")
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        print(f"Error downloading {filepath}: {e}")
        raise e

# Download dataset if not present
if not os.path.exists("nerf_real_360"):
    zip_path = "nerf_real_360.zip"
    download_file(
        "https://huggingface.co/datasets/1kaiser/NERF_360/resolve/main/nerf_real_360.zip?download=true", 
        zip_path
    )
    print("Extracting nerf_real_360.zip...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(".")
    print("Extraction complete.")
else:
    print("Dataset directory nerf_real_360/ already exists.")

# Download models if not present
download_file(
    "https://huggingface.co/datasets/1kaiser/vggt-omega-jax/resolve/main/vggt_omega_1b_512.pt", 
    "vggt_omega_1b_512.pt"
)
download_file(
    "https://huggingface.co/datasets/1kaiser/vggt-omega-jax/resolve/main/vggt_omega_1b_512.msgpack.zst", 
    "vggt_omega_1b_512.msgpack.zst"
)

# %% [markdown]
# ## 1. Load and Preprocess Images
# Load 2 images from the nerf real 360 pinecone dataset.

# %%
image_dir = "nerf_real_360/pinecone/images"
image_paths = sorted(glob.glob(os.path.join(image_dir, "*")))[:2]
print("Loading images:")
for p in image_paths:
    print(f"  - {os.path.basename(p)}")

# PyTorch preprocessing -> shape [B, T, C, H, W]
x_pt = load_and_preprocess_images(image_paths, image_resolution=512, patch_size=16)
x_pt = x_pt.unsqueeze(0)
print(f"Preprocessed PyTorch shape: {x_pt.shape}")

# JAX preprocessing -> shape [B, T, H, W, C]
x_jax = jnp.array(x_pt.permute(0, 1, 3, 4, 2).numpy())
print(f"Preprocessed JAX shape: {x_jax.shape}")

# %% [markdown]
# ## 2. PyTorch Inference

# %%
print("Initializing PyTorch model...")
pt_model = PTModel(
    patch_size=16,
    embed_dim=1024,
    enable_camera=True,
    enable_depth=True,
    enable_alignment=False
).eval()

pt_state_dict = torch.load("vggt_omega_1b_512.pt", map_location="cpu")
if "model" in pt_state_dict:
    pt_state_dict = pt_state_dict["model"]
pt_model.load_state_dict(pt_state_dict, strict=True)

print("Running PyTorch CPU Inference...")
t0 = time.time()
with torch.no_grad():
    pt_preds = pt_model(x_pt)
pt_time = time.time() - t0
print(f"PyTorch CPU inference completed in {pt_time:.4f} seconds")

# %% [markdown]
# ## 3. JAX Inference

# %%
print("Initializing JAX model...")
jax_model = JAXModel(
    patch_size=16,
    embed_dim=1024,
    enable_camera=True,
    enable_depth=True,
    enable_alignment=False
)

variables_template = jax_model.init(jax.random.PRNGKey(0), jnp.zeros((1, len(image_paths), 512, 512, 3)))
restored_params = load_checkpoint(variables_template, "vggt_omega_1b_512.msgpack.zst")

@jax.jit
def jax_predict(params, x):
    return jax_model.apply(params, x)

print("Compiling JAX model (first run)...")
t0 = time.time()
jax_preds = jax_predict(restored_params, x_jax)
for k, v in jax_preds.items():
    if isinstance(v, jnp.ndarray):
        v.block_until_ready()
jax_compile_time = time.time() - t0
print(f"JAX CPU first run (compile + inference) completed in {jax_compile_time:.4f} seconds")

print("Running JAX compiled inference (second run)...")
t0 = time.time()
jax_preds = jax_predict(restored_params, x_jax)
for k, v in jax_preds.items():
    if isinstance(v, jnp.ndarray):
        v.block_until_ready()
jax_jit_time = time.time() - t0
print(f"JAX CPU compiled inference completed in {jax_jit_time:.4f} seconds")

# %% [markdown]
# ## 4. Output Parity Verification

# %%
print("--- Verifying Output Parity ---")
keys = ["camera_and_register_tokens", "pose_enc", "depth", "depth_conf"]
all_passed = True

for key in keys:
    pt_val = pt_preds[key]
    if isinstance(pt_val, torch.Tensor):
        pt_val = pt_val.cpu().numpy()
    jax_val = np.array(jax_preds[key])
    
    diff = np.max(np.abs(pt_val - jax_val))
    mean_diff = np.mean(np.abs(pt_val - jax_val))
    
    status = "PASSED" if diff < 1e-3 else "FAILED"
    if status == "FAILED":
        all_passed = False
        
    print(f"{key}:")
    print(f"  Shape: {pt_val.shape}")
    print(f"  Max Absolute Difference: {diff:.6e}")
    print(f"  Mean Absolute Difference: {mean_diff:.6e}")
    print(f"  Status: {status}")

if all_passed:
    print("\nSUCCESS: All outputs match within the 1e-3 threshold!")
else:
    print("\nFAIL: Parity mismatch detected!")

# %% [markdown]
# ## 5. Visualize Results

# %%
fig, axes = plt.subplots(2, 4, figsize=(18, 9))

for i in range(2):
    # Original Image
    img_np = (x_jax[0, i] * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])).clip(0, 1)
    axes[i, 0].imshow(img_np)
    axes[i, 0].set_title(f"Frame {i} Image")
    axes[i, 0].axis("off")
    
    # PyTorch Depth
    im_pt = axes[i, 1].imshow(pt_preds["depth"][0, i, ..., 0].cpu().numpy(), cmap="inferno")
    axes[i, 1].set_title(f"PyTorch Depth (Frame {i})")
    axes[i, 1].axis("off")
    fig.colorbar(im_pt, ax=axes[i, 1], fraction=0.046, pad=0.04)
    
    # JAX Depth
    im_jax = axes[i, 2].imshow(jax_preds["depth"][0, i, ..., 0], cmap="inferno")
    axes[i, 2].set_title(f"JAX Depth (Frame {i})")
    axes[i, 2].axis("off")
    fig.colorbar(im_jax, ax=axes[i, 2], fraction=0.046, pad=0.04)
    
    # Difference Map
    diff_depth = np.abs(pt_preds["depth"][0, i, ..., 0].cpu().numpy() - jax_preds["depth"][0, i, ..., 0])
    im_diff = axes[i, 3].imshow(diff_depth, cmap="coolwarm")
    axes[i, 3].set_title(f"Absolute Diff (Frame {i})")
    axes[i, 3].axis("off")
    fig.colorbar(im_diff, ax=axes[i, 3], fraction=0.046, pad=0.04)

plt.suptitle("PyTorch vs JAX Depth Comparison on Pinecone Dataset", fontsize=16)
plt.tight_layout()
plt.savefig("parity_comparison.png", dpi=150)
plt.show()
print("Saved comparison plot to parity_comparison.png")

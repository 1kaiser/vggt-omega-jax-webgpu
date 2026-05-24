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
from vggt_omega.utils.pose_enc import encoding_to_camera

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

def unproject_depth_map_to_point_map(depth_map: np.ndarray, extrinsic: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    depth = depth_map[..., 0]
    num_frames, height, width = depth.shape

    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    x = np.broadcast_to(x[None], (num_frames, height, width))
    y = np.broadcast_to(y[None], (num_frames, height, width))

    fx = intrinsic[:, 0, 0][:, None, None]
    fy = intrinsic[:, 1, 1][:, None, None]
    cx = intrinsic[:, 0, 2][:, None, None]
    cy = intrinsic[:, 1, 2][:, None, None]

    camera_points = np.stack(
        [
            (x - cx) / fx * depth,
            (y - cy) / fy * depth,
            depth,
        ],
        axis=-1,
    )

    rotation = extrinsic[:, :3, :3]
    translation = extrinsic[:, :3, 3]
    return np.einsum(
        "sij,shwj->shwi",
        np.transpose(rotation, (0, 2, 1)),
        camera_points - translation[:, None, None, :],
    )

# Download dataset if not present
if not os.path.exists("pinecone"):
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
    print("Dataset directory pinecone/ already exists.")

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
image_dir = "pinecone/images"
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
# ## 2. Benchmarking Runner (Subprocess Execution)
# To measure the peak memory of each model configuration accurately without memory pollution, we run them in separate processes.

# %%
import sys
import subprocess
import gc
import resource

# Check if running inside a Jupyter notebook:
is_notebook = 'ipykernel' in sys.modules

if not is_notebook:
    import argparse
    parser = argparse.ArgumentParser(description="VGGT-Omega Parity and Subprocess Benchmark")
    parser.add_argument("--mode", type=str, choices=["run_all", "pytorch", "jax_fp32", "jax_bf16_jit", "jax_mmap"], default="run_all")
    args, unknown = parser.parse_known_args()
    mode = args.mode
else:
    mode = "run_all"

def get_peak_ram_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

if mode == "pytorch":
    print("--- Running PyTorch CPU Baseline (float32) ---")
    print(f"Start RAM: {get_peak_ram_mb():.2f} MB")
    
    t_start = time.time()
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
    t_load = time.time() - t_start
    print(f"Weights loaded in {t_load:.2f} s. RAM: {get_peak_ram_mb():.2f} MB")
    
    t0 = time.time()
    with torch.no_grad():
        pt_preds = pt_model(x_pt)
    t_inf = time.time() - t0
    
    # Save outputs to file
    np.savez(
        "pt_preds.npz",
        camera_and_register_tokens=pt_preds["camera_and_register_tokens"].cpu().numpy(),
        pose_enc=pt_preds["pose_enc"].cpu().numpy(),
        depth=pt_preds["depth"].cpu().numpy(),
        depth_conf=pt_preds["depth_conf"].cpu().numpy(),
        images=pt_preds["images"].cpu().numpy()
    )
    
    print(f"BENCHMARK_METRICS: load_s={t_load:.4f}, compile_s=0.0000, inf_s={t_inf:.4f}, peak_ram_mb={get_peak_ram_mb():.2f}")
    sys.exit(0)

elif mode == "jax_fp32":
    print("--- Running JAX CPU Baseline (float32, with template init) ---")
    print(f"Start RAM: {get_peak_ram_mb():.2f} MB")
    
    t_start = time.time()
    jax_model = JAXModel(
        patch_size=16,
        embed_dim=1024,
        enable_camera=True,
        enable_depth=True,
        enable_alignment=False
    )
    
    variables_template = jax_model.init(jax.random.PRNGKey(0), jnp.zeros((1, len(image_paths), 512, 512, 3)))
    restored_params = load_checkpoint(variables_template, "vggt_omega_1b_512.msgpack.zst")
    t_load = time.time() - t_start
    print(f"Weights loaded in {t_load:.2f} s. RAM: {get_peak_ram_mb():.2f} MB")
    
    @jax.jit
    def jax_predict(params, x):
        return jax_model.apply(params, x)
        
    t0 = time.time()
    preds = jax_predict(restored_params, x_jax)
    for k, v in preds.items():
        if isinstance(v, jnp.ndarray):
            v.block_until_ready()
    t_compile = time.time() - t0
    
    t0 = time.time()
    preds = jax_predict(restored_params, x_jax)
    for k, v in preds.items():
        if isinstance(v, jnp.ndarray):
            v.block_until_ready()
    t_inf = time.time() - t0
    
    np.savez(
        "jax_preds.npz",
        camera_and_register_tokens=np.array(preds["camera_and_register_tokens"]).astype(np.float32),
        pose_enc=np.array(preds["pose_enc"]).astype(np.float32),
        depth=np.array(preds["depth"]).astype(np.float32),
        depth_conf=np.array(preds["depth_conf"]).astype(np.float32)
    )
    print(f"BENCHMARK_METRICS: load_s={t_load:.4f}, compile_s={t_compile:.4f}, inf_s={t_inf:.4f}, peak_ram_mb={get_peak_ram_mb():.2f}")
    sys.exit(0)

elif mode == "jax_bf16_jit":
    print("--- Running JAX CPU Low-RAM (bfloat16, direct loading, JIT) ---")
    print(f"Start RAM: {get_peak_ram_mb():.2f} MB")
    
    t_start = time.time()
    # Initialize in bfloat16
    jax_model = JAXModel(
        patch_size=16,
        embed_dim=1024,
        enable_camera=True,
        enable_depth=True,
        enable_alignment=False,
        dtype=jnp.bfloat16
    )
    
    # Bypass template initialization to save RAM
    restored_params = load_checkpoint(None, "vggt_omega_1b_512_bf16.msgpack.zst")
    t_load = time.time() - t_start
    print(f"Weights loaded in {t_load:.2f} s. RAM: {get_peak_ram_mb():.2f} MB")
    
    # Cast input to bfloat16
    x_jax_bf16 = x_jax.astype(jnp.bfloat16)
    
    @jax.jit
    def jax_predict(params, x):
        return jax_model.apply(params, x)
        
    t0 = time.time()
    preds = jax_predict(restored_params, x_jax_bf16)
    for k, v in preds.items():
        if isinstance(v, jnp.ndarray):
            v.block_until_ready()
    t_compile = time.time() - t0
    
    t0 = time.time()
    preds = jax_predict(restored_params, x_jax_bf16)
    for k, v in preds.items():
        if isinstance(v, jnp.ndarray):
            v.block_until_ready()
    t_inf = time.time() - t0
    
    np.savez(
        "jax_bf16_preds.npz",
        camera_and_register_tokens=np.array(preds["camera_and_register_tokens"]).astype(np.float32),
        pose_enc=np.array(preds["pose_enc"]).astype(np.float32),
        depth=np.array(preds["depth"]).astype(np.float32),
        depth_conf=np.array(preds["depth_conf"]).astype(np.float32)
    )
    
    print(f"BENCHMARK_METRICS: load_s={t_load:.4f}, compile_s={t_compile:.4f}, inf_s={t_inf:.4f}, peak_ram_mb={get_peak_ram_mb():.2f}")
    sys.exit(0)

elif mode == "jax_mmap":
    print("--- Running JAX CPU Ultra-Low-RAM (float16, memory-mapped, eager) ---")
    print(f"Start RAM: {get_peak_ram_mb():.2f} MB")
    
    t_start = time.time()
    
    # Increase open files limit
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (32768, hard))
    except Exception as e:
        print(f"Warning: could not increase open file limit: {e}")
        
    # Initialize in float16
    jax_model = JAXModel(
        patch_size=16,
        embed_dim=1024,
        enable_camera=True,
        enable_depth=True,
        enable_alignment=False,
        dtype=jnp.float16
    )
    
    # Load memory-mapped parameters from directory
    def load_recursive(current_dir):
        d = {}
        for entry in os.scandir(current_dir):
            if entry.is_dir():
                d[entry.name] = load_recursive(entry.path)
            elif entry.is_file() and entry.name.endswith(".npy"):
                key = entry.name[:-4]
                d[key] = np.load(entry.path, mmap_mode='r')
        return d
        
    restored_params = load_recursive("vggt_omega_1b_512_fp16_mmap")
    t_load = time.time() - t_start
    print(f"Weights loaded in {t_load:.2f} s. RAM: {get_peak_ram_mb():.2f} MB")
    
    # Cast input to float16 and disable JIT
    x_jax_fp16 = x_jax.astype(jnp.float16)
    
    t0 = time.time()
    preds = jax_model.apply(restored_params, x_jax_fp16)
    for k, v in preds.items():
        if isinstance(v, jnp.ndarray):
            v.block_until_ready()
    t_inf = time.time() - t0
    
    np.savez(
        "jax_mmap_preds.npz",
        camera_and_register_tokens=np.array(preds["camera_and_register_tokens"]).astype(np.float32),
        pose_enc=np.array(preds["pose_enc"]).astype(np.float32),
        depth=np.array(preds["depth"]).astype(np.float32),
        depth_conf=np.array(preds["depth_conf"]).astype(np.float32)
    )
    
    print(f"BENCHMARK_METRICS: load_s={t_load:.4f}, compile_s=0.0000, inf_s={t_inf:.4f}, peak_ram_mb={get_peak_ram_mb():.2f}")
    sys.exit(0)

# Parent runner logic (run_all mode):
print("Executing subprocess benchmarks...")
python_bin = sys.executable
benchmarks = {}

for target_mode in ["pytorch", "jax_fp32", "jax_bf16_jit", "jax_mmap"]:
    print(f"\nLaunching {target_mode} subprocess...")
    try:
        script_path = __file__
    except NameError:
        script_path = "inference_comparison.py"
    cmd = [python_bin, script_path, "--mode", target_mode]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Subprocess failed with code {res.returncode}")
        print("Stdout:", res.stdout)
        print("Stderr:", res.stderr)
        raise RuntimeError(f"Benchmark failed for {target_mode}")
        
    print(res.stdout)
    
    # Parse metrics
    metrics_line = [line for line in res.stdout.splitlines() if "BENCHMARK_METRICS" in line]
    if metrics_line:
        metrics_str = metrics_line[0].split("BENCHMARK_METRICS: ")[1]
        metrics = dict(item.split("=") for item in metrics_str.split(", "))
        benchmarks[target_mode] = {k: float(v) for k, v in metrics.items()}

# Load saved predictions for parity and visualization
def load_and_cast_preds(path):
    data = np.load(path)
    d = {}
    for k in data.files:
        val = data[k]
        if val.dtype.kind in ['f', 'b', 'V', 'O']:
            try:
                d[k] = val.astype(np.float32)
            except Exception:
                d[k] = val
        else:
            d[k] = val
    return d

pt_preds = load_and_cast_preds("pt_preds.npz")
jax_preds = load_and_cast_preds("jax_preds.npz")
jax_bf16_preds = load_and_cast_preds("jax_bf16_preds.npz")
jax_mmap_preds = load_and_cast_preds("jax_mmap_preds.npz")

# %% [markdown]
# ## 4. Output Parity Verification

# %%
print("--- Verifying Output Parity ---")
keys = ["camera_and_register_tokens", "pose_enc", "depth", "depth_conf"]

configs = [
    ("JAX Baseline (fp32)", jax_preds),
    ("JAX Low-RAM (bf16)", jax_bf16_preds),
    ("JAX Ultra-Low-RAM (fp16 mmap)", jax_mmap_preds)
]

for name, preds in configs:
    print(f"\nComparing PyTorch vs {name}:")
    all_passed = True
    for key in keys:
        pt_val = pt_preds[key]
        jax_val = preds[key]
        
        diff = np.max(np.abs(pt_val - jax_val))
        mean_diff = np.mean(np.abs(pt_val - jax_val))
        
        # bf16/fp16 has slightly higher numeric tolerance (e.g. 5e-3 / 1e-3)
        tol = 5e-3 if "bf16" in name else 1e-3
        status = "PASSED" if diff < tol else "FAILED"
        if status == "FAILED":
            all_passed = False
            
        print(f"  {key}:")
        print(f"    Shape: {pt_val.shape}")
        print(f"    Max Absolute Difference: {diff:.6e}")
        print(f"    Mean Absolute Difference: {mean_diff:.6e}")
        print(f"    Status: {status}")

# Print final Comparison Table
print("\n" + "="*80)
print("                    VGGT-OMEGA INFERENCE PERFORMANCE COMPARISON")
print("="*80)
print(f"| Model / Implementation | Precision | Mode | Loading Time | Warm Inference | Peak RAM |")
print(f"| :--- | :--- | :--- | :---: | :---: | :---: |")
print(f"| PyTorch CPU Baseline | float32 | Eager | {benchmarks['pytorch']['load_s']:.2f} s | {benchmarks['pytorch']['inf_s']:.4f} s | {benchmarks['pytorch']['peak_ram_mb']:.1f} MB |")
print(f"| JAX CPU Baseline | float32 | JIT | {benchmarks['jax_fp32']['load_s']:.2f} s | {benchmarks['jax_fp32']['inf_s']:.4f} s | {benchmarks['jax_fp32']['peak_ram_mb']:.1f} MB |")
print(f"| JAX CPU Low-RAM | bfloat16 | JIT | {benchmarks['jax_bf16_jit']['load_s']:.2f} s | {benchmarks['jax_bf16_jit']['inf_s']:.4f} s | {benchmarks['jax_bf16_jit']['peak_ram_mb']:.1f} MB |")
print(f"| JAX CPU Ultra-Low-RAM | float16 | Eager (mmap) | {benchmarks['jax_mmap']['load_s']:.2f} s | {benchmarks['jax_mmap']['inf_s']:.4f} s | {benchmarks['jax_mmap']['peak_ram_mb']:.1f} MB |")
print("="*80)

# %% [markdown]
# ## 5. Visualize Results

# %%
model_names = [
    "PyTorch CPU Baseline (float32)",
    "JAX CPU Baseline (float32)",
    "JAX CPU Low-RAM (bfloat16)",
    "JAX CPU Ultra-Low-RAM (float16 mmap)"
]
preds_list = [pt_preds, jax_preds, jax_bf16_preds, jax_mmap_preds]

fig, axes = plt.subplots(4, 4, figsize=(18, 16))

for r in range(4):
    model_name = model_names[r]
    preds = preds_list[r]
    
    for i in range(2):
        # Column 2*i: Original Image
        img_np = (x_jax[0, i] * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])).clip(0, 1)
        axes[r, 2 * i].imshow(img_np)
        axes[r, 2 * i].set_title(f"{model_name}\nFrame {i} Image")
        axes[r, 2 * i].axis("off")
        
        # Column 2*i+1: Depth Map
        im = axes[r, 2 * i + 1].imshow(preds["depth"][0, i, ..., 0], cmap="inferno")
        axes[r, 2 * i + 1].set_title(f"{model_name}\nFrame {i} Depth")
        axes[r, 2 * i + 1].axis("off")
        fig.colorbar(im, ax=axes[r, 2 * i + 1], fraction=0.046, pad=0.04)

plt.suptitle("PyTorch vs Converted JAX Models Depth Parity Comparison (Pinecone Dataset)", fontsize=18, fontweight="bold")
plt.tight_layout()
plt.savefig("parity_comparison.png", dpi=150)
plt.show()
print("Saved comparison plot to parity_comparison.png")

# %% [markdown]
# ## 6. 3D Reconstruction Orthographic View Comparison

# %%
print("Processing 3D point cloud projections...")

# Preprocess colors (PyTorch format -> channels_last [0, 1])
pt_images_np = pt_preds["images"][0]
colors_np = np.transpose(pt_images_np, (0, 2, 3, 1))
colors_np = (colors_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])).clip(0, 1)

def process_run(world_points, extrinsics, confidence, colors):
    T = extrinsics.shape[0]
    ext4x4 = np.zeros((T, 4, 4))
    ext4x4[:, :3, :4] = extrinsics
    ext4x4[:, 3, 3] = 1.0
    
    R0 = ext4x4[0, :3, :3]
    t0 = ext4x4[0, :3, 3]
    
    # Transform points to Cam0 frame: P_c0 = R0 * P_w + t0
    points_c0 = np.einsum("ij,thwj->thwi", R0, world_points) + t0[None, None, None, :]
    
    # Transform camera positions to Cam0 frame
    cam_centers_c0 = []
    cam_dirs_c0 = []
    for i in range(T):
        Ri = ext4x4[i, :3, :3]
        ti = ext4x4[i, :3, 3]
        Cw = -Ri.T @ ti
        Cc0 = R0 @ Cw + t0
        
        Di_w = Ri[2, :] # 3rd row of Ri (OpenCV optical axis Z-axis)
        Di_c0 = R0 @ Di_w
        
        cam_centers_c0.append(Cc0)
        cam_dirs_c0.append(Di_c0)
        
    cam_centers_c0 = np.array(cam_centers_c0)
    cam_dirs_c0 = np.array(cam_dirs_c0)
    
    # Filter point cloud using confidence
    flat_points = points_c0.reshape(-1, 3)
    flat_conf = confidence.reshape(-1)
    flat_colors = colors.reshape(-1, 3)
    
    valid_mask = np.isfinite(flat_points).all(axis=-1)
    flat_points = flat_points[valid_mask]
    flat_conf = flat_conf[valid_mask]
    flat_colors = flat_colors[valid_mask]
    
    if len(flat_conf) > 0:
        thres = np.percentile(flat_conf, 85)
        mask = flat_conf >= thres
        flat_points = flat_points[mask]
        flat_colors = flat_colors[mask]
        
    if len(flat_points) > 10000:
        idx = np.random.choice(len(flat_points), 10000, replace=False)
        flat_points = flat_points[idx]
        flat_colors = flat_colors[idx]
        
    return flat_points, flat_colors, cam_centers_c0, cam_dirs_c0

def get_model_3d_data(preds, colors_np, pt_preds):
    pose_enc_torch = torch.from_numpy(np.array(preds["pose_enc"]))
    extrinsic, intrinsic = encoding_to_camera(
        pose_enc_torch,
        pt_preds["images"].shape[-2:],
    )
    extrinsic_np = extrinsic.numpy()[0]
    intrinsic_np = intrinsic.numpy()[0]
    depth_np = np.array(preds["depth"])[0]
    conf_np = np.array(preds["depth_conf"])[0]
    
    world_points = unproject_depth_map_to_point_map(depth_np, extrinsic_np, intrinsic_np)
    return process_run(world_points, extrinsic_np, conf_np, colors_np)

pt_pts, pt_clrs, pt_cams, pt_dirs = get_model_3d_data(pt_preds, colors_np, pt_preds)
jax_pts, jax_clrs, jax_cams, jax_dirs = get_model_3d_data(jax_preds, colors_np, pt_preds)
jax_bf16_pts, jax_bf16_clrs, jax_bf16_cams, jax_bf16_dirs = get_model_3d_data(jax_bf16_preds, colors_np, pt_preds)
jax_mmap_pts, jax_mmap_clrs, jax_mmap_cams, jax_mmap_dirs = get_model_3d_data(jax_mmap_preds, colors_np, pt_preds)

fig = plt.figure(figsize=(24, 24))

model_labels = [
    "PyTorch CPU Baseline (float32)",
    "JAX CPU Baseline (float32)",
    "JAX CPU Low-RAM (bfloat16)",
    "JAX CPU Ultra-Low-RAM (float16 mmap)"
]

model_data = [
    (pt_pts, pt_clrs, pt_cams, pt_dirs),
    (jax_pts, jax_clrs, jax_cams, jax_dirs),
    (jax_bf16_pts, jax_bf16_clrs, jax_bf16_cams, jax_bf16_dirs),
    (jax_mmap_pts, jax_mmap_clrs, jax_mmap_cams, jax_mmap_dirs)
]

# Find overall limits to align all views properly
all_pts = [d[0] for d in model_data if len(d[0]) > 0]
if len(all_pts) > 0:
    min_x = min(pts[:, 0].min() for pts in all_pts) - 0.2
    max_x = max(pts[:, 0].max() for pts in all_pts) + 0.2
    min_y_val = min(pts[:, 1].min() for pts in all_pts) - 0.2
    max_y_val = max(pts[:, 1].max() for pts in all_pts) + 0.2
    min_z = min(pts[:, 2].min() for pts in all_pts) - 0.2
    max_z = max(pts[:, 2].max() for pts in all_pts) + 0.2
else:
    min_x, max_x = -1.0, 1.0
    min_y_val, max_y_val = -1.0, 1.0
    min_z, max_z = -1.0, 1.0

min_h = -max_y_val
max_h = -min_y_val

T = len(pt_cams)
cmap_cam = plt.cm.rainbow(np.linspace(0, 1, T))

for row in range(4):
    label = model_labels[row]
    pts, clrs, cams, dirs = model_data[row]
    
    # Column 1: Top View (X vs Z)
    ax1 = fig.add_subplot(4, 4, row * 4 + 1)
    if len(pts) > 0:
        ax1.scatter(pts[:, 0], pts[:, 2], c=clrs, s=2, alpha=0.6)
    for idx in range(T):
        c_pos = cams[idx]
        c_dir = dirs[idx]
        c_color = cmap_cam[idx]
        ax1.scatter(c_pos[0], c_pos[2], color=c_color, marker="^", s=100, edgecolors="black", label=f"Camera {idx}" if row==0 else "")
        ax1.plot([c_pos[0], c_pos[0] + 0.15 * c_dir[0]], [c_pos[2], c_pos[2] + 0.15 * c_dir[2]], color=c_color, linewidth=2)
    ax1.set_title(f"{label}\nTop View (X vs Z)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("X (Left-Right)")
    ax1.set_ylabel("Z (Forward-Back)")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.set_xlim(min_x, max_x)
    ax1.set_ylim(min_z, max_z)
    if row == 0:
        ax1.legend()
        
    # Column 2: Side View (Z vs Y)
    ax2 = fig.add_subplot(4, 4, row * 4 + 2)
    if len(pts) > 0:
        ax2.scatter(pts[:, 2], pts[:, 1], c=clrs, s=2, alpha=0.6)
    for idx in range(T):
        c_pos = cams[idx]
        c_dir = dirs[idx]
        c_color = cmap_cam[idx]
        ax2.scatter(c_pos[2], c_pos[1], color=c_color, marker="^", s=100, edgecolors="black")
        ax2.plot([c_pos[2], c_pos[2] + 0.15 * c_dir[2]], [c_pos[1], c_pos[1] + 0.15 * c_dir[1]], color=c_color, linewidth=2)
    ax2.set_title(f"{label}\nSide View (Z vs Y)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Z (Forward-Back)")
    ax2.set_ylabel("Y (Up-Down)")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.set_xlim(min_z, max_z)
    ax2.set_ylim(min_y_val, max_y_val)
    ax2.invert_yaxis()
    
    # Column 3: Front View (X vs Y)
    ax3 = fig.add_subplot(4, 4, row * 4 + 3)
    if len(pts) > 0:
        ax3.scatter(pts[:, 0], pts[:, 1], c=clrs, s=2, alpha=0.6)
    for idx in range(T):
        c_pos = cams[idx]
        c_dir = dirs[idx]
        c_color = cmap_cam[idx]
        ax3.scatter(c_pos[0], c_pos[1], color=c_color, marker="^", s=100, edgecolors="black")
        ax3.plot([c_pos[0], c_pos[0] + 0.15 * c_dir[0]], [c_pos[1], c_pos[1] + 0.15 * c_dir[1]], color=c_color, linewidth=2)
    ax3.set_title(f"{label}\nFront View (X vs Y)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("X (Left-Right)")
    ax3.set_ylabel("Y (Up-Down)")
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.set_xlim(min_x, max_x)
    ax3.set_ylim(min_y_val, max_y_val)
    ax3.invert_yaxis()
    
    # Column 4: 3D Isometric View (X, Z, -Y)
    ax4 = fig.add_subplot(4, 4, row * 4 + 4, projection='3d')
    if len(pts) > 0:
        ax4.scatter(pts[:, 0], pts[:, 2], -pts[:, 1], c=clrs, s=2, alpha=0.6)
    for idx in range(T):
        c_pos = cams[idx]
        c_dir = dirs[idx]
        c_color = cmap_cam[idx]
        ax4.scatter(c_pos[0], c_pos[2], -c_pos[1], color=c_color, marker="^", s=100, edgecolors="black")
        ax4.plot(
            [c_pos[0], c_pos[0] + 0.15 * c_dir[0]],
            [c_pos[2], c_pos[2] + 0.15 * c_dir[2]],
            [-c_pos[1], -c_pos[1] - 0.15 * c_dir[1]],
            color=c_color,
            linewidth=2
        )
    ax4.set_title(f"{label}\n3D Isometric View", fontsize=11, fontweight="bold")
    ax4.set_xlabel("X (Left-Right)")
    ax4.set_ylabel("Z (Forward-Back)")
    ax4.set_zlabel("Height (-Y)")
    ax4.set_xlim(min_x, max_x)
    ax4.set_ylim(min_z, max_z)
    ax4.set_zlim(min_h, max_h)
    ax4.view_init(elev=30, azim=45)
    ax4.grid(True, linestyle="--", alpha=0.5)

plt.suptitle("3D Reconstruction Multi-View Comparison\nPyTorch vs JAX (Vanilla, Low-RAM, Ultra-Low-RAM) on Pinecone Dataset", fontsize=18, fontweight="bold")
plt.tight_layout()
plt.savefig("views_comparison.png", dpi=150)
plt.show()
print("Saved 3D views comparison to views_comparison.png")


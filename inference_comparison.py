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
        camera_and_register_tokens=np.array(preds["camera_and_register_tokens"]),
        pose_enc=np.array(preds["pose_enc"]),
        depth=np.array(preds["depth"]),
        depth_conf=np.array(preds["depth_conf"])
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
pt_preds = np.load("pt_preds.npz")
jax_preds = np.load("jax_preds.npz")

# %% [markdown]
# ## 4. Output Parity Verification

# %%
print("--- Verifying Output Parity ---")
keys = ["camera_and_register_tokens", "pose_enc", "depth", "depth_conf"]
all_passed = True

for key in keys:
    pt_val = pt_preds[key]
    jax_val = jax_preds[key]
    
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
num_plot_frames = min(4, len(image_paths))
fig, axes = plt.subplots(num_plot_frames, 4, figsize=(18, 4.5 * num_plot_frames))

if num_plot_frames == 1:
    axes = np.expand_dims(axes, axis=0)

for i in range(num_plot_frames):
    # Original Image
    img_np = (x_jax[0, i] * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])).clip(0, 1)
    axes[i, 0].imshow(img_np)
    axes[i, 0].set_title(f"Frame {i} Image")
    axes[i, 0].axis("off")
    
    # PyTorch Depth
    im_pt = axes[i, 1].imshow(pt_preds["depth"][0, i, ..., 0], cmap="inferno")
    axes[i, 1].set_title(f"PyTorch Depth (Frame {i})")
    axes[i, 1].axis("off")
    fig.colorbar(im_pt, ax=axes[i, 1], fraction=0.046, pad=0.04)
    
    # JAX Depth
    im_jax = axes[i, 2].imshow(jax_preds["depth"][0, i, ..., 0], cmap="inferno")
    axes[i, 2].set_title(f"JAX Depth (Frame {i})")
    axes[i, 2].axis("off")
    fig.colorbar(im_jax, ax=axes[i, 2], fraction=0.046, pad=0.04)
    
    # Difference Map
    diff_depth = np.abs(pt_preds["depth"][0, i, ..., 0] - jax_preds["depth"][0, i, ..., 0])
    im_diff = axes[i, 3].imshow(diff_depth, cmap="coolwarm")
    axes[i, 3].set_title(f"Absolute Diff (Frame {i})")
    axes[i, 3].axis("off")
    fig.colorbar(im_diff, ax=axes[i, 3], fraction=0.046, pad=0.04)

plt.suptitle(f"PyTorch vs JAX Depth Comparison on Pinecone Dataset (First {num_plot_frames} Frames)", fontsize=16)
plt.tight_layout()
plt.savefig("parity_comparison.png", dpi=150)
plt.show()
print("Saved comparison plot to parity_comparison.png")

# %% [markdown]
# ## 6. 3D Reconstruction Orthographic View Comparison

# %%
print("Processing 3D point cloud projections...")

# Decode camera parameters
pt_extrinsic, pt_intrinsic = encoding_to_camera(
    torch.from_numpy(pt_preds["pose_enc"]),
    pt_preds["images"].shape[-2:],
)
pt_extrinsic_np = pt_extrinsic.numpy()[0]
pt_intrinsic_np = pt_intrinsic.numpy()[0]
pt_depth_np = pt_preds["depth"][0]
pt_conf_np = pt_preds["depth_conf"][0]
pt_images_np = pt_preds["images"][0]

pt_world_points = unproject_depth_map_to_point_map(pt_depth_np, pt_extrinsic_np, pt_intrinsic_np)

# Decode JAX camera parameters (convert pose_enc from numpy/jax array to torch tensor first)
jax_pose_enc_torch = torch.from_numpy(np.array(jax_preds["pose_enc"]))
jax_extrinsic, jax_intrinsic = encoding_to_camera(
    jax_pose_enc_torch,
    pt_preds["images"].shape[-2:],
)
jax_extrinsic_np = jax_extrinsic.numpy()[0]
jax_intrinsic_np = jax_intrinsic.numpy()[0]
jax_depth_np = np.array(jax_preds["depth"])[0]
jax_conf_np = np.array(jax_preds["depth_conf"])[0]

jax_world_points = unproject_depth_map_to_point_map(jax_depth_np, jax_extrinsic_np, jax_intrinsic_np)

# Preprocess colors (PyTorch format -> channels_last [0, 1])
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

pt_pts, pt_clrs, pt_cams, pt_dirs = process_run(pt_world_points, pt_extrinsic_np, pt_conf_np, colors_np)
jax_pts, jax_clrs, jax_cams, jax_dirs = process_run(jax_world_points, jax_extrinsic_np, jax_conf_np, colors_np)

fig, axes = plt.subplots(3, 2, figsize=(14, 18))

views = [
    ("Top View (X vs Z)", 0, 2, "X (Left-Right)", "Z (Forward-Back)", False),
    ("Side View (Z vs Y)", 2, 1, "Z (Forward-Back)", "Y (Up-Down)", True),
    ("Front View (X vs Y)", 0, 1, "X (Left-Right)", "Y (Up-Down)", True)
]

T = len(pt_cams)
cmap_cam = plt.cm.rainbow(np.linspace(0, 1, T))

for row, (view_title, d1, d2, l1, l2, inv_y) in enumerate(views):
    # PyTorch Column
    ax_pt = axes[row, 0]
    ax_pt.scatter(pt_pts[:, d1], pt_pts[:, d2], c=pt_clrs, s=2, alpha=0.6)
    
    for idx in range(T):
        c_pos = pt_cams[idx]
        c_dir = pt_dirs[idx]
        c_color = cmap_cam[idx]
        ax_pt.scatter(c_pos[d1], c_pos[d2], color=c_color, marker="^", s=100, edgecolors="black", label=f"Camera {idx}" if row==0 else "")
        ax_pt.plot([c_pos[d1], c_pos[d1] + 0.15 * c_dir[d1]], [c_pos[d2], c_pos[d2] + 0.15 * c_dir[d2]], color=c_color, linewidth=2)
        
    ax_pt.set_title(f"PyTorch - {view_title}", fontsize=12, fontweight="bold")
    ax_pt.set_xlabel(l1)
    ax_pt.set_ylabel(l2)
    ax_pt.grid(True, linestyle="--", alpha=0.5)
    if inv_y:
        ax_pt.invert_yaxis()
    if row == 0:
        ax_pt.legend()
        
    # JAX Column
    ax_jax = axes[row, 1]
    ax_jax.scatter(jax_pts[:, d1], jax_pts[:, d2], c=jax_clrs, s=2, alpha=0.6)
    
    for idx in range(T):
        c_pos = jax_cams[idx]
        c_dir = jax_dirs[idx]
        c_color = cmap_cam[idx]
        ax_jax.scatter(c_pos[d1], c_pos[d2], color=c_color, marker="^", s=100, edgecolors="black", label=f"Camera {idx}" if row==0 else "")
        ax_jax.plot([c_pos[d1], c_pos[d1] + 0.15 * c_dir[d1]], [c_pos[d2], c_pos[d2] + 0.15 * c_dir[d2]], color=c_color, linewidth=2)
        
    ax_jax.set_title(f"JAX - {view_title}", fontsize=12, fontweight="bold")
    ax_jax.set_xlabel(l1)
    ax_jax.set_ylabel(l2)
    ax_jax.grid(True, linestyle="--", alpha=0.5)
    if inv_y:
        ax_jax.invert_yaxis()
    if row == 0:
        ax_jax.legend()
        
    min_d1 = min(pt_pts[:, d1].min(), jax_pts[:, d1].min()) - 0.2
    max_d1 = max(pt_pts[:, d1].max(), jax_pts[:, d1].max()) + 0.2
    min_d2 = min(pt_pts[:, d2].min(), jax_pts[:, d2].min()) - 0.2
    max_d2 = max(pt_pts[:, d2].max(), jax_pts[:, d2].max()) + 0.2
    
    ax_pt.set_xlim(min_d1, max_d1)
    ax_pt.set_ylim(min_d2, max_d2)
    ax_jax.set_xlim(min_d1, max_d1)
    ax_jax.set_ylim(min_d2, max_d2)

plt.suptitle("3D Reconstruction Orthographic View Comparison\nPyTorch (Original) vs JAX Port on Pinecone Dataset", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig("views_comparison.png", dpi=150)
plt.show()
print("Saved 3D views comparison to views_comparison.png")

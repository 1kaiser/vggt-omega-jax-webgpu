import os
import sys

# Memory optimization: restrict glibc memory arenas to prevent RSS bloat on multi-core systems
if os.environ.get("MALLOC_ARENA_MAX") != "1":
    os.environ["MALLOC_ARENA_MAX"] = "1"
    try:
        os.execv(sys.executable, [sys.executable] + sys.argv)
    except Exception as e:
        print(f"Warning: Failed to set MALLOC_ARENA_MAX=1 via re-exec: {e}")

import glob
import time
import argparse
import numpy as np
import resource

# Force JAX to use CPU backend and enable multi-threading
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["OMP_NUM_THREADS"] = "32"
os.environ["MKL_NUM_THREADS"] = "32"
os.environ["OPENBLAS_NUM_THREADS"] = "32"
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=true"

import jax
import jax.numpy as jnp
import ml_dtypes

from vggt_omega.jax.models import VGGTOmega as JAXModel

def get_peak_ram_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

def run_profile(num_frames, weights_dir, resolution=512):
    print(f"\n==================================================")
    print(f"Profiling VGGT-Omega with {num_frames} frames (Eager Mode)")
    print(f"==================================================")
    
    # 1. Reset memory usage metrics tracking (if possible) or track delta
    ram_start = get_peak_ram_mb()
    print(f"Baseline RAM: {ram_start:.2f} MB")
    
    # 2. Load images
    image_dir = "nerf_real_360/pinecone/images"
    from vggt_omega.utils.load_fn import load_and_preprocess_images_np
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*")))[:num_frames]
    
    print(f"Loading and preprocessing {num_frames} images...")
    t_data_0 = time.time()
    x_np = load_and_preprocess_images_np(image_paths, image_resolution=resolution, patch_size=16)
    x_np = np.expand_dims(x_np, axis=0) # add batch dim [1, num_frames, 3, H, W]
    x_jax = jnp.array(np.transpose(x_np, (0, 1, 3, 4, 2)), dtype=jnp.bfloat16)
    t_data = time.time() - t_data_0
    print(f"Data ready. Shape: {x_jax.shape}. Time: {t_data:.2f}s")
    
    ram_after_data = get_peak_ram_mb()
    print(f"RAM after data load: {ram_after_data:.2f} MB (Delta: {ram_after_data - ram_start:.2f} MB)")
    
    # 3. Init Model
    jax_model = JAXModel(
        patch_size=16,
        embed_dim=1024,
        enable_camera=True,
        enable_depth=True,
        enable_alignment=False,
        dtype=jnp.bfloat16
    )
    
    # 4. Load weights
    print(f"Loading memory-mapped parameters from {weights_dir}...")
    t_load_0 = time.time()
    def load_recursive(current_dir):
        d = {}
        for entry in os.scandir(current_dir):
            if entry.is_dir():
                d[entry.name] = load_recursive(entry.path)
            elif entry.is_file() and entry.name.endswith(".npy"):
                key = entry.name[:-4]
                val = np.load(entry.path, mmap_mode='r')
                if val.dtype == np.dtype('V2') or val.dtype.kind == 'V':
                    d[key] = val.view(ml_dtypes.bfloat16)
                else:
                    d[key] = val
        return d
    restored_params = load_recursive(weights_dir)
    t_load = time.time() - t_load_0
    print(f"Weights loaded in {t_load:.2f}s")
    
    ram_after_weights = get_peak_ram_mb()
    print(f"RAM after weights load: {ram_after_weights:.2f} MB (Delta from start: {ram_after_weights - ram_start:.2f} MB)")
    
    # 5. Eager Inference
    print("Running eager inference (process 2 frames at a time)...")
    t_inf_0 = time.time()
    
    # Track CPU utilization during inference using a simple cpu times check
    cpu_start_times = os.times()
    
    preds_list = []
    for start_idx in range(0, x_jax.shape[1], 2):
        x_batch = x_jax[:, start_idx : start_idx + 2]
        batch_preds = jax_model.apply(restored_params, x_batch)
        for k, v in batch_preds.items():
            if isinstance(v, jnp.ndarray):
                v.block_until_ready()
        preds_list.append(batch_preds)
        
    t_inf = time.time() - t_inf_0
    cpu_end_times = os.times()
    
    # Compute user + system CPU time
    cpu_time = (cpu_end_times.user - cpu_start_times.user) + (cpu_end_times.system - cpu_start_times.system)
    cpu_utilization = (cpu_time / t_inf) * 100 if t_inf > 0 else 0.0
    
    ram_end = get_peak_ram_mb()
    print(f"Inference completed in {t_inf:.2f}s")
    print(f"Peak RAM during inference: {ram_end:.2f} MB (Delta from start: {ram_end - ram_start:.2f} MB)")
    
    fps = num_frames / t_inf if t_inf > 0 else 0.0
    print(f"Performance: {fps:.2f} FPS")
    print(f"Approx. CPU Utilization: {cpu_utilization:.1f}%")
    
    return {
        "num_frames": num_frames,
        "load_time_s": t_load,
        "inf_time_s": t_inf,
        "peak_ram_mb": ram_end,
        "ram_delta_mb": ram_end - ram_start,
        "fps": fps,
        "cpu_util_pct": cpu_utilization
    }

def main():
    # Raise open file descriptors limit
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (32768, hard))
    except Exception as e:
        print(f"Warning: could not increase open file limit: {e}")

    parser = argparse.ArgumentParser(description="Profile VGGT-Omega performance on 2, 4, and 6 frames.")
    parser.add_argument("--weights", type=str, default="vggt_omega_1b_512_bf16_mmap", help="Path to weights directory.")
    args = parser.parse_args()

    # Allow block-level JIT compilation
    # jax.config.update('jax_disable_jit', True)

    results = []
    for f in [2, 4, 6]:
        # Run in subprocess or clean up variables manually to prevent RAM accumulation from previous runs
        # Since we want accurate RAM delta per run, running them in a single process might accumulate memory,
        # but we can track delta since start, which shows the growth pattern.
        res = run_profile(f, args.weights)
        results.append(res)

    print("\n\n==================================================")
    print("PROFILING SUMMARY TABLE")
    print("==================================================")
    print(f"{'Frames':<8} | {'Inf Time (s)':<12} | {'FPS':<8} | {'Peak RAM (MB)':<14} | {'RAM Delta (MB)':<15} | {'CPU Util (%)':<12}")
    print("-" * 80)
    for r in results:
        print(f"{r['num_frames']:<8} | {r['inf_time_s']:<12.2f} | {r['fps']:<8.2f} | {r['peak_ram_mb']:<14.2f} | {r['ram_delta_mb']:<15.2f} | {r['cpu_util_pct']:<12.1f}")
    print("==================================================")

if __name__ == "__main__":
    main()

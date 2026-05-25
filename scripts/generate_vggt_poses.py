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
import json
import time
import argparse
import numpy as np
import torch

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
from vggt_omega.utils.pose_enc import encoding_to_camera

def get_peak_ram_mb():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

def main():
    import resource
    # Raise open file descriptors limit to avoid "Too many open files" with mmap
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (32768, hard))
    except Exception as e:
        print(f"Warning: could not increase open file limit: {e}")

    parser = argparse.ArgumentParser(description="Generate camera poses using VGGT-Omega JAX in ultra-low RAM mode.")
    parser.add_argument("--weights", type=str, required=True, help="Path to weights directory.")
    parser.add_argument("--dataset-dir", type=str, default="nerf_real_360/pinecone", help="Path to scene directory.")
    parser.add_argument("--resolution", type=int, default=512, help="Resolution width/height.")
    parser.add_argument("--num-frames", type=int, default=0, help="Number of frames to process (0 for all).")
    parser.add_argument("--disable-jit", action="store_true", help="Disable JIT compilation for eager execution.")
    args = parser.parse_args()

    if args.disable_jit:
        print("Disabling JIT compilation for eager execution...")
        jax.config.update('jax_disable_jit', True)
    else:
        print("Enabling JIT compilation...")

    image_dir = os.path.join(args.dataset_dir, "images")
    if not os.path.exists(image_dir):
        raise ValueError(f"Dataset path {image_dir} not found.")

    from vggt_omega.utils.load_fn import load_and_preprocess_images
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*")))
    if args.num_frames > 0:
        image_paths = image_paths[:args.num_frames]
    num_frames = len(image_paths)
    print(f"Found {num_frames} images. Loading and preprocessing...")

    # Load and preprocess all images
    # We preprocess them and stack them
    x_pt = load_and_preprocess_images(image_paths, image_resolution=args.resolution, patch_size=16)
    # x_pt shape is [num_frames, 3, H, W]
    x_pt = x_pt.unsqueeze(0) # [1, num_frames, 3, H, W]
    x_jax = jnp.array(x_pt.permute(0, 1, 3, 4, 2).numpy(), dtype=jnp.bfloat16)

    print(f"Input shape: {x_jax.shape}, dtype: {x_jax.dtype}")

    # Initialize JAX model in bfloat16
    print("Initializing VGGTOmega JAX model in bfloat16...")
    jax_model = JAXModel(
        patch_size=16,
        embed_dim=1024,
        enable_camera=True,
        enable_depth=True,
        enable_alignment=False,
        dtype=jnp.bfloat16
    )

    # Load memory-mapped parameters from directory
    print(f"Loading memory-mapped parameters from directory {args.weights}...")
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
    restored_params = load_recursive(args.weights)

    def jax_predict(params, x):
        return jax_model.apply(params, x)

    print("Running inference (JIT compilation enabled)..." if not args.disable_jit else "Running eager inference...")
    t0 = time.time()
    preds_list = []
    # Process 2 frames at a time to avoid memory spikes
    for start_idx in range(0, x_jax.shape[1], 2):
        x_batch = x_jax[:, start_idx : start_idx + 2]
        if args.disable_jit:
            batch_preds = jax_model.apply(restored_params, x_batch)
        else:
            batch_preds = jax_predict(restored_params, x_batch)
        
        # Block until ready
        for k, v in batch_preds.items():
            if isinstance(v, jnp.ndarray):
                v.block_until_ready()
                
        batch_np = {
            "pose_enc": np.array(batch_preds["pose_enc"]).astype(np.float32),
        }
        preds_list.append(batch_np)
    t_inf = time.time() - t0
    print(f"Inference completed in {t_inf:.2f} seconds.")

    # Concatenate all poses
    pose_enc = np.concatenate([p["pose_enc"] for p in preds_list], axis=1) # [1, num_frames, 9]

    # Convert to PyTorch tensor for decoding
    pose_enc_pt = torch.from_numpy(pose_enc)

    # Get input image size (actual resolution after loading)
    # The default load size in load_and_preprocess_images is balanced, yielding actual shape
    # Let's get the shape from x_pt
    H_actual, W_actual = x_pt.shape[-2:]
    print(f"Decoded image shape: {H_actual}x{W_actual}")

    # Decode pose_enc to camera parameters
    print("Decoding poses to camera extrinsics and intrinsics...")
    extrinsics_pt, intrinsics_pt = encoding_to_camera(pose_enc_pt, (H_actual, W_actual))
    
    extrinsics = extrinsics_pt.squeeze(0).numpy() # [num_frames, 3, 4]
    intrinsics = intrinsics_pt.squeeze(0).numpy() # [num_frames, 3, 3]

    print("Converting poses from OpenCV (world-to-camera) to OpenGL (camera-to-world)...")
    frames = []
    
    # Calculate average intrinsics
    fx_avg = np.mean(intrinsics[:, 0, 0])
    fy_avg = np.mean(intrinsics[:, 1, 1])
    cx_avg = np.mean(intrinsics[:, 0, 2])
    cy_avg = np.mean(intrinsics[:, 1, 2])

    for idx, image_path in enumerate(image_paths):
        # 1. World-to-camera [3, 4]
        extri = extrinsics[idx] # [3, 4]
        # Make it homogeneous [4, 4]
        world_to_cam = np.eye(4)
        world_to_cam[:3] = extri
        
        # 2. Invert to get camera-to-world (OpenCV)
        cam_to_world_opencv = np.linalg.inv(world_to_cam)
        
        # 3. Convert from OpenCV to OpenGL convention
        # Y-down -> Y-up, Z-forwards -> Z-backwards
        flip_yz = np.diag([1, -1, -1, 1])
        cam_to_world_opengl = cam_to_world_opencv @ flip_yz
        
        # Format the relative file path as required by load_blender_posedata
        rel_path = os.path.join("images", os.path.basename(image_path))
        
        frames.append({
            "file_path": rel_path,
            "transform_matrix": cam_to_world_opengl.tolist()
        })

    # Build transforms.json structure
    transforms = {
        "w": int(W_actual),
        "h": int(H_actual),
        "fl_x": float(fx_avg),
        "fl_y": float(fy_avg),
        "cx": float(cx_avg),
        "cy": float(cy_avg),
        "frames": frames
    }

    out_file = os.path.join(args.dataset_dir, "transforms.json")
    print(f"Saving transforms to {out_file}...")
    with open(out_file, "w") as f:
        json.dump(transforms, f, indent=2)

    print("Success! Camera poses exported successfully.")

if __name__ == "__main__":
    main()

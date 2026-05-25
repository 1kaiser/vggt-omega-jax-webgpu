import os
import glob
import time
import gc
import resource
import argparse
import numpy as np

# Force JAX to use CPU backend (since host GPU driver mismatch prevents JAX CUDA init)
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import ml_dtypes

from vggt_omega.jax.models import VGGTOmega as JAXModel
from vggt_omega.jax.load_weights import load_checkpoint

def get_peak_ram_mb():
    # Returns peak RSS in MB
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

def main():
    # Raise open file descriptors limit to avoid "Too many open files" with mmap
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (32768, hard))
    except Exception as e:
        print(f"Warning: could not increase open file limit: {e}")

    parser = argparse.ArgumentParser(description="Run VGGT-Omega JAX Low RAM inference.")
    parser.add_argument("--weights", type=str, required=True, help="Path to .msgpack.zst weights file.")
    parser.add_argument("--dtype", type=str, choices=["float32", "float16", "bfloat16"], default="bfloat16", help="Model precision.")
    parser.add_argument("--disable-jit", action="store_true", help="Disable JIT compilation for eager execution.")
    parser.add_argument("--no-template", action="store_true", help="Bypass model parameter template initialization.")
    parser.add_argument("--num-frames", type=int, default=2, help="Number of frames to run inference on.")
    parser.add_argument("--resolution", type=int, default=512, help="Image resolution width/height.")
    args = parser.parse_args()

    if args.disable_jit:
        print("Disabling JIT compilation for eager execution...")
        jax.config.update('jax_disable_jit', True)

    print(f"Peak RAM before loading anything: {get_peak_ram_mb():.2f} MB")

    # Determine dtypes
    if args.dtype == "float16":
        jnp_dtype = jnp.float16
    elif args.dtype == "bfloat16":
        jnp_dtype = jnp.bfloat16
    else:
        jnp_dtype = jnp.float32

    # Load images from pinecone dataset
    image_dir = "nerf_real_360/pinecone/images"
    if not os.path.exists(image_dir):
        print(f"Dataset path {image_dir} not found. Running with dummy input.")
        x_jax = jnp.zeros((1, args.num_frames, args.resolution, args.resolution, 3), dtype=jnp_dtype)
    else:
        from vggt_omega.utils.load_fn import load_and_preprocess_images
        image_paths = sorted(glob.glob(os.path.join(image_dir, "*")))[:args.num_frames]
        print(f"Loading {len(image_paths)} images...")
        x_pt = load_and_preprocess_images(image_paths, image_resolution=args.resolution, patch_size=16)
        x_pt = x_pt.unsqueeze(0) # add batch dim
        x_jax = jnp.array(x_pt.permute(0, 1, 3, 4, 2).numpy(), dtype=jnp_dtype)

    print(f"Input shape: {x_jax.shape}, dtype: {x_jax.dtype}")
    print(f"Peak RAM after loading input: {get_peak_ram_mb():.2f} MB")

    # Initialize JAX model
    print(f"Initializing VGGTOmega JAX model in {args.dtype}...")
    t_start = time.time()
    jax_model = JAXModel(
        patch_size=16,
        embed_dim=1024,
        enable_camera=True,
        enable_depth=True,
        enable_alignment=False,
        dtype=jnp_dtype
    )

    if args.no_template:
        print("Bypassing parameter template initialization to save RAM...")
        variables_template = None
    else:
        # Initialize template parameters directly in target dtype
        print("Initializing parameter template...")
        dummy_img = jnp.zeros((1, args.num_frames, args.resolution, args.resolution, 3), dtype=jnp_dtype)
        variables_template = jax_model.init(jax.random.PRNGKey(0), dummy_img)
        print(f"Peak RAM after template init: {get_peak_ram_mb():.2f} MB")

    # Load weights
    if os.path.isdir(args.weights):
        print(f"Loading memory-mapped parameters from directory {args.weights}...")
        def load_recursive(current_dir):
            d = {}
            for entry in os.scandir(current_dir):
                if entry.is_dir():
                    d[entry.name] = load_recursive(entry.path)
                elif entry.is_file() and entry.name.endswith(".npy"):
                    key = entry.name[:-4]
                    d[key] = np.load(entry.path, mmap_mode='r')
            return d
        restored_params = load_recursive(args.weights)
    else:
        print(f"Loading checkpoint {args.weights}...")
        restored_params = load_checkpoint(variables_template, args.weights)
    
    # Delete template to free up RAM
    if variables_template is not None:
        del variables_template
        gc.collect()

    print(f"Peak RAM after weights loaded: {get_peak_ram_mb():.2f} MB")

    @jax.jit
    def jax_predict(params, x):
        return jax_model.apply(params, x)

    print("Running inference...")
    t0 = time.time()
    if args.disable_jit:
        jax_preds = jax_model.apply(restored_params, x_jax)
    else:
        jax_preds = jax_predict(restored_params, x_jax)
    
    # Force evaluation of JAX arrays to get correct execution time and peak memory
    for k, v in jax_preds.items():
        if isinstance(v, jnp.ndarray):
            v.block_until_ready()
    t_inf = time.time() - t0
    
    print(f"Inference completed in {t_inf:.4f} seconds")
    print(f"Total time elapsed: {time.time() - t_start:.2f} seconds")
    print(f"Peak RAM during/after inference: {get_peak_ram_mb():.2f} MB")

    # Print output shapes
    for k, v in jax_preds.items():
        if isinstance(v, jnp.ndarray):
            print(f"Output '{k}': shape {v.shape}, dtype {v.dtype}")

if __name__ == "__main__":
    main()

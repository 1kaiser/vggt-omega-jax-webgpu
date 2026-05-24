import os
import argparse
import flax.serialization
import zstandard as zstd
import numpy as np

def save_recursive(d, current_dir):
    os.makedirs(current_dir, exist_ok=True)
    for k, v in d.items():
        if isinstance(v, dict):
            save_recursive(v, os.path.join(current_dir, k))
        else:
            # Convert ml_dtypes.bfloat16 or other custom JAX arrays to standard numpy array representation
            if hasattr(v, "dtype") and str(v.dtype) == "bfloat16":
                # Convert bfloat16 to raw bytes representation or save as np.ndarray (numpy supports ml_dtypes if imported)
                pass
            filepath = os.path.join(current_dir, f"{k}.npy")
            # Save using standard numpy save
            np.save(filepath, np.asarray(v))

def main():
    parser = argparse.ArgumentParser(description="Convert compressed msgpack.zst weights to folder of mmap npy files.")
    parser.add_argument("--weights", type=str, required=True, help="Path to msgpack.zst weights file.")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory for npy files.")
    args = parser.parse_args()

    print(f"Loading checkpoint {args.weights}...")
    with open(args.weights, "rb") as f:
        compressed_bytes = f.read()
        
    dctx = zstd.ZstdDecompressor()
    serialized_bytes = dctx.decompress(compressed_bytes)
    
    print("Deserializing parameters...")
    params = flax.serialization.msgpack_restore(serialized_bytes)
    
    print(f"Saving params recursively to {args.out_dir}...")
    save_recursive(params, args.out_dir)
    print("Done! Parameters successfully exported as memory-mapped npy files.")

if __name__ == "__main__":
    main()

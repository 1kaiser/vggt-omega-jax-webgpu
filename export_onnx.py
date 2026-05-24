import os
import sys
import torch
import warnings

# Avoid warnings
warnings.filterwarnings("ignore")

from vggt_omega.models import VGGTOmega

def test_onnx_export(embed_dim=256, resolution=256, output_path="vggt_omega_small.onnx"):
    print(f"Initializing VGGT-Omega model (embed_dim={embed_dim}, resolution={resolution})...")
    
    # Instantiate the model with random weights
    model = VGGTOmega(
        patch_size=16,
        embed_dim=embed_dim,
        enable_camera=True,
        enable_depth=True,
        enable_alignment=False
    )
    model.eval()
    
    # Create dummy input of shape [B, T, C, H, W]
    B, T, C, H, W = 1, 2, 3, resolution, resolution
    dummy_input = torch.randn(B, T, C, H, W)
    
    print(f"Dummy input shape: {dummy_input.shape}")
    
    # Perform ONNX export
    print(f"Exporting to {output_path} via torch.onnx.export...")
    
    # We specify opset_version=17 for maximum compatibility with modern operators like meshgrid/tile
    try:
        torch.onnx.export(
            model,
            (dummy_input,),
            output_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["images"],
            output_names=["camera_and_register_tokens", "pose_enc", "depth", "depth_conf"],
            dynamo=False
        )
        print(f"SUCCESS: Exported model to {output_path} successfully!")
        
        # Check model file size
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"ONNX Model size: {size_mb:.2f} MB")
        return True
    except Exception as e:
        print(f"FAILURE: Export failed with error: {e}")
        return False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed-dim", type=int, default=256, help="Embedding dimension for the test model (e.g. 256 or 1024)")
    parser.add_argument("--resolution", type=int, default=256, help="Input resolution (height and width)")
    parser.add_argument("--output", type=str, default="vggt_omega_test.onnx", help="Output path for the ONNX file")
    args = parser.parse_args()
    
    success = test_onnx_export(embed_dim=args.embed_dim, resolution=args.resolution, output_path=args.output)
    sys.exit(0 if success else 1)

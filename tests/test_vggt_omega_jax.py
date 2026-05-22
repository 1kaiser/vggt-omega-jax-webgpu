import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pytest
import torch
import jax
import jax.numpy as jnp
import numpy as np

from vggt_omega.models import VGGTOmega as PTModel
from vggt_omega.jax.models import VGGTOmega as JAXModel
from vggt_omega.jax.convert_weights import map_key, convert_tensor, insert_at_path

def test_vggt_omega_parity():
    # Set seeds for reproducibility and numerical stability of random weights
    torch.manual_seed(0)
    np.random.seed(0)
    
    # 1. Initialize PyTorch model with a small embed_dim to speed up test and save memory
    pt_model = PTModel(
        patch_size=16,
        embed_dim=256,
        enable_camera=True,
        enable_depth=True,
        enable_alignment=True
    ).eval()
    
    # Initialize the bias_mask buffers to avoid NaNs from random initialization
    for module in pt_model.modules():
        if hasattr(module, "bias_mask") and module.bias_mask is not None:
            o = module.out_features
            module.bias_mask.fill_(1.0)
            module.bias_mask[o // 3 : 2 * o // 3].fill_(0.0)
            
    # 2. Extract PyTorch state dict and convert to Flax params format
    pt_state_dict = pt_model.state_dict()
    jax_params = {}
    for pt_key, pt_tensor in pt_state_dict.items():
        if "_resnet_" in pt_key or "bias_mask" in pt_key:
            continue
        jax_path = map_key(pt_key)
        val = convert_tensor(pt_tensor, jax_path)
        insert_at_path(jax_params, jax_path, val)
        
    wrapped_params = {"params": jax_params}
    
    # 3. Initialize JAX model with the same configuration
    jax_model = JAXModel(
        patch_size=16,
        embed_dim=256,
        enable_camera=True,
        enable_depth=True,
        enable_alignment=True
    )
    
    # Create random input: batch_size=1, num_frames=2, channels=3, height=64, width=64
    np.random.seed(42)
    dummy_input_np = np.random.randn(1, 2, 3, 64, 64).astype(np.float32)
    
    # PyTorch input: [B, T, C, H, W]
    x_pt = torch.from_numpy(dummy_input_np)
    
    # JAX input: [B, T, H, W, C]
    x_jax = jnp.array(np.transpose(dummy_input_np, (0, 1, 3, 4, 2)))
    
    # 4. Run PyTorch forward pass
    with torch.no_grad():
        pt_preds = pt_model(x_pt)
        
    # 5. Run JAX forward pass
    # We pass the converted parameters directly as variables
    jax_preds = jax_model.apply(wrapped_params, x_jax)
    
    # 6. Verify shapes and values
    # Compare "camera_and_register_tokens"
    pt_tokens = pt_preds["camera_and_register_tokens"].cpu().numpy()
    jax_tokens = np.array(jax_preds["camera_and_register_tokens"])
    print(f"PT tokens contains NaN: {np.isnan(pt_tokens).any()}")
    print(f"JAX tokens contains NaN: {np.isnan(jax_tokens).any()}")
    print("PT tokens stats: min={:.4e}, max={:.4e}, mean={:.4e}, std={:.4e}".format(
        pt_tokens.min(), pt_tokens.max(), pt_tokens.mean(), pt_tokens.std()
    ))
    print("JAX tokens stats: min={:.4e}, max={:.4e}, mean={:.4e}, std={:.4e}".format(
        jax_tokens.min(), jax_tokens.max(), jax_tokens.mean(), jax_tokens.std()
    ))
    print("PT tokens slice (first 5 elements of first token):", pt_tokens[0, 0, 0, :5])
    print("JAX tokens slice (first 5 elements of first token):", jax_tokens[0, 0, 0, :5])
    
    assert pt_tokens.shape == jax_tokens.shape, f"Tokens shape mismatch: {pt_tokens.shape} vs {jax_tokens.shape}"
    diff_tokens = np.max(np.abs(pt_tokens - jax_tokens))
    print(f"camera_and_register_tokens max diff: {diff_tokens:.2e}")
    assert diff_tokens < 1e-5, f"Tokens mismatch: max diff is {diff_tokens:.2e}"
    
    # Compare "pose_enc"
    pt_pose = pt_preds["pose_enc"].cpu().numpy()
    jax_pose = np.array(jax_preds["pose_enc"])
    assert pt_pose.shape == jax_pose.shape, f"Pose shape mismatch: {pt_pose.shape} vs {jax_pose.shape}"
    diff_pose = np.max(np.abs(pt_pose - jax_pose))
    print(f"pose_enc max diff: {diff_pose:.2e}")
    assert diff_pose < 1e-5, f"Pose mismatch: max diff is {diff_pose:.2e}"
    
    # Compare "depth"
    pt_depth = pt_preds["depth"].cpu().numpy()
    jax_depth = np.array(jax_preds["depth"])
    assert pt_depth.shape == jax_depth.shape, f"Depth shape mismatch: {pt_depth.shape} vs {jax_depth.shape}"
    diff_depth = np.max(np.abs(pt_depth - jax_depth))
    print(f"depth max diff: {diff_depth:.2e}")
    assert diff_depth < 1e-5, f"Depth mismatch: max diff is {diff_depth:.2e}"
    
    # Compare "depth_conf"
    pt_conf = pt_preds["depth_conf"].cpu().numpy()
    jax_conf = np.array(jax_preds["depth_conf"])
    assert pt_conf.shape == jax_conf.shape, f"Confidence shape mismatch: {pt_conf.shape} vs {jax_conf.shape}"
    diff_conf = np.max(np.abs(pt_conf - jax_conf))
    print(f"depth_conf max diff: {diff_conf:.2e}")
    assert diff_conf < 1e-5, f"Confidence mismatch: max diff is {diff_conf:.2e}"
    
    # Compare "text_alignment_embedding"
    pt_align = pt_preds["text_alignment_embedding"].cpu().numpy()
    jax_align = np.array(jax_preds["text_alignment_embedding"])
    assert pt_align.shape == jax_align.shape, f"Alignment shape mismatch: {pt_align.shape} vs {jax_align.shape}"
    diff_align = np.max(np.abs(pt_align - jax_align))
    print(f"text_alignment_embedding max diff: {diff_align:.2e}")
    assert diff_align < 1e-5, f"Alignment mismatch: max diff is {diff_align:.2e}"
    
    print("All checks passed successfully!")

if __name__ == "__main__":
    test_vggt_omega_parity()

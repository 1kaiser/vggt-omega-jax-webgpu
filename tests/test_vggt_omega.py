import pytest
import torch
from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera


def test_vggt_omega_init():
    """Test model initialization with various configurations."""
    model = VGGTOmega(
        patch_size=16,
        embed_dim=256,  # Use smaller embed_dim to speed up test execution / save memory
        enable_camera=True,
        enable_depth=True,
        enable_alignment=True,
    )
    assert model.camera_head is not None
    assert model.dense_head is not None
    assert model.text_alignment_head is not None


def test_vggt_omega_forward_cuda():
    """Test forward pass of the model on a dummy input tensor using CUDA."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available, skipping CUDA forward pass test")

    device = torch.device("cuda")
    
    # Initialize a small test model to avoid out of memory and slow initialization
    model = VGGTOmega(
        patch_size=16,
        embed_dim=256,
        enable_camera=True,
        enable_depth=True,
        enable_alignment=True,
    ).to(device).eval()

    # Create dummy video input of shape [batch_size, num_frames, channels, height, width]
    # Height and width must be divisible by patch_size (16)
    dummy_images = torch.randn(1, 2, 3, 128, 128, device=device)

    with torch.inference_mode():
        predictions = model(dummy_images)

    assert isinstance(predictions, dict)
    assert "camera_and_register_tokens" in predictions
    assert "pose_enc" in predictions
    assert "depth" in predictions
    assert "depth_conf" in predictions
    assert "text_alignment_embedding" in predictions

    # Check predicted shape matches expected
    # Dense head outputs depth of shape [batch_size, num_frames, height, width, 1]
    assert predictions["depth"].shape == (1, 2, 128, 128, 1)
    assert predictions["depth_conf"].shape == (1, 2, 128, 128)


def test_vggt_omega_utils():
    """Test utility functions in vggt_omega.utils."""
    # Test encoding_to_camera utility
    # pose_encoding expects batch dimensions, typically [batch_size, num_frames, 9]
    dummy_pose_enc = torch.randn(1, 2, 9) # [batch_size, num_frames, 9] (translation (3), quaternion (4), FoV (2))
    image_shape = (128, 128)
    extrinsics, intrinsics = encoding_to_camera(dummy_pose_enc, image_shape)
    
    # Expected shapes: extrinsics is [batch_size, num_frames, 3, 4], intrinsics is [batch_size, num_frames, 3, 3]
    assert extrinsics.shape == (1, 2, 3, 4)
    assert intrinsics.shape == (1, 2, 3, 3)

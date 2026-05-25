# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import List, Tuple, Dict, Any, Callable
import jax
import jax.numpy as jnp
import flax.linen as nn
import ctypes
import gc

try:
    libc = ctypes.CDLL("libc.so.6")
except Exception:
    libc = None

def trim_memory():
    gc.collect()
    if libc is not None:
        try:
            libc.malloc_trim(0)
        except Exception:
            pass

from vggt_omega.jax.layers import (
    RMSNorm,
    LayerScale,
    PatchEmbed,
    RopePositionEmbedding,
    SelfAttentionBlock,
)


def bilinear_interpolate(x: jnp.ndarray, size: Tuple[int, int]) -> jnp.ndarray:
    # x shape: [B, H, W, C]
    # size: (H_out, W_out)
    B, H_in, W_in, C = x.shape
    H_out, W_out = size
    if H_in == H_out and W_in == W_out:
        return x
    y_out = jnp.arange(H_out, dtype=jnp.float32)
    x_out = jnp.arange(W_out, dtype=jnp.float32)

    # align_corners coordinate mapping
    y_in = y_out * (H_in - 1) / (H_out - 1)
    x_in = x_out * (W_in - 1) / (W_out - 1)

    y0 = jnp.floor(y_in).astype(jnp.int32)
    y1 = jnp.minimum(y0 + 1, H_in - 1)
    wy = y_in - y0

    x0 = jnp.floor(x_in).astype(jnp.int32)
    x1 = jnp.minimum(x0 + 1, W_in - 1)
    wx = x_in - x0

    v00 = x[:, y0, :, :][:, :, x0, :]
    v01 = x[:, y0, :, :][:, :, x1, :]
    v10 = x[:, y1, :, :][:, :, x0, :]
    v11 = x[:, y1, :, :][:, :, x1, :]

    wy_grid = wy[None, :, None, None]
    wx_grid = wx[None, None, :, None]

    return (1 - wy_grid) * ((1 - wx_grid) * v00 + wx_grid * v01) + wy_grid * ((1 - wx_grid) * v10 + wx_grid * v11)


def pixel_shuffle(x: jnp.ndarray, r: int) -> jnp.ndarray:
    # x shape: [B, H, W, C_in] where C_in = C_out * r * r
    B, H, W, C_in = x.shape
    C_out = C_in // (r * r)
    x = jnp.reshape(x, (B, H, W, C_out, r, r))
    x = jnp.transpose(x, (0, 1, 4, 2, 5, 3))
    return jnp.reshape(x, (B, H * r, W * r, C_out))


def create_uv_grid(
    width: int, height: int, aspect_ratio: float | None = None, dtype: jnp.dtype = jnp.float32
) -> jnp.ndarray:
    if aspect_ratio is None:
        aspect_ratio = float(width) / float(height)

    diag_factor = (aspect_ratio**2 + 1.0) ** 0.5
    span_x = aspect_ratio / diag_factor
    span_y = 1.0 / diag_factor

    left_x = -span_x * (width - 1) / width
    right_x = span_x * (width - 1) / width
    top_y = -span_y * (height - 1) / height
    bottom_y = span_y * (height - 1) / height

    x_coords = jnp.linspace(left_x, right_x, num=width, dtype=dtype)
    y_coords = jnp.linspace(top_y, bottom_y, num=height, dtype=dtype)

    uu, vv = jnp.meshgrid(x_coords, y_coords, indexing="xy")
    uv_grid = jnp.stack((uu, vv), axis=-1)

    return uv_grid


def make_sincos_pos_embed(embed_dim: int, pos: jnp.ndarray, omega_0: float = 100.0) -> jnp.ndarray:
    assert embed_dim % 2 == 0
    omega = jnp.arange(embed_dim // 2, dtype=jnp.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / (omega_0**omega)

    pos = jnp.reshape(pos, -1)
    out = jnp.einsum("m,d->md", pos, omega)

    emb_sin = jnp.sin(out)
    emb_cos = jnp.cos(out)

    emb = jnp.concatenate([emb_sin, emb_cos], axis=1)
    return emb.astype(jnp.float32)


def position_grid_to_embed(pos_grid: jnp.ndarray, embed_dim: int, omega_0: float = 100.0) -> jnp.ndarray:
    H, W, grid_dim = pos_grid.shape
    assert grid_dim == 2
    pos_flat = jnp.reshape(pos_grid, (-1, grid_dim))

    emb_x = make_sincos_pos_embed(embed_dim // 2, pos_flat[:, 0], omega_0=omega_0)
    emb_y = make_sincos_pos_embed(embed_dim // 2, pos_flat[:, 1], omega_0=omega_0)

    emb = jnp.concatenate([emb_x, emb_y], axis=-1)
    return jnp.reshape(emb, (H, W, embed_dim))


class DinoVisionTransformer(nn.Module):
    patch_size: int = 16
    embed_dim: int = 1024
    depth: int = 24
    num_heads: int = 16
    ffn_ratio: float = 4.0
    qkv_bias: bool = True
    proj_bias: bool = True
    ffn_bias: bool = True
    n_storage_tokens: int = 4
    layerscale_init: float | None = 1e-5
    norm_layer: str = "layernormbf16"
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.patch_embed = PatchEmbed(
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            flatten_embedding=False,
            dtype=self.dtype,
            name="patch_embed",
        )
        self.cls_token = self.param("cls_token", nn.initializers.normal(stddev=0.02), (1, 1, self.embed_dim))
        if self.n_storage_tokens > 0:
            self.storage_tokens = self.param(
                "storage_tokens", nn.initializers.normal(stddev=0.02), (1, self.n_storage_tokens, self.embed_dim)
            )
        self.mask_token = self.param("mask_token", nn.initializers.zeros, (1, self.embed_dim))

        self.rope_embed = RopePositionEmbedding(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            base=100.0,
            normalize_coords="max",
            dtype=self.dtype,
            name="rope_embed",
        )

        self.blocks = [
            SelfAttentionBlock(
                dim=self.embed_dim,
                num_heads=self.num_heads,
                ffn_ratio=self.ffn_ratio,
                qkv_bias=self.qkv_bias,
                proj_bias=self.proj_bias,
                ffn_bias=self.ffn_bias,
                init_values=self.layerscale_init,
                norm_layer="rmsnorm" if self.norm_layer == "rmsnorm" else "layernorm",
                dtype=self.dtype,
                name=f"blocks_{i}",
            )
            for i in range(self.depth)
        ]

        if self.norm_layer == "rmsnorm":
            self.norm = RMSNorm(dim=self.embed_dim, dtype=self.dtype, name="norm")
        else:
            self.norm = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype, name="norm")

    def __call__(self, x: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        # Input shape: [B, H, W, C]
        x = self.patch_embed(x)
        B, Hp, Wp, _ = x.shape
        x = jnp.reshape(x, (B, Hp * Wp, self.embed_dim))

        cls_token = jnp.broadcast_to(self.cls_token, (B, 1, self.embed_dim))
        if self.n_storage_tokens > 0:
            storage_tokens = jnp.broadcast_to(self.storage_tokens, (B, self.n_storage_tokens, self.embed_dim))
            x = jnp.concatenate([cls_token, storage_tokens, x], axis=1)
        else:
            x = jnp.concatenate([cls_token, x], axis=1)

        rope_sin, rope_cos = self.rope_embed(Hp, Wp)
        rope = (rope_sin, rope_cos)

        for blk in self.blocks:
            x = blk(x, rope=rope)
            x.block_until_ready()
            trim_memory()

        x_norm = self.norm(x)
        x_norm_cls_reg = x_norm[:, : self.n_storage_tokens + 1]
        x_norm_patch = x_norm[:, self.n_storage_tokens + 1 :]

        return {
            "x_norm_clstoken": x_norm_cls_reg[:, 0],
            "x_storage_tokens": x_norm_cls_reg[:, 1:],
            "x_norm_patchtokens": x_norm_patch,
            "x_prenorm": x,
        }


def slice_expand_and_flatten(token_tensor: jnp.ndarray, batch_size: int, num_frames: int) -> jnp.ndarray:
    # token_tensor: [1, 2, K, embed_dim]
    first_frame_token = token_tensor[:, 0:1]  # [1, 1, K, embed_dim]
    first_frame_token = jnp.broadcast_to(first_frame_token, (batch_size, 1, *token_tensor.shape[2:]))

    if num_frames == 1:
        tokens = first_frame_token
    else:
        other_frame_tokens = token_tensor[:, 1:]
        other_frame_tokens = jnp.broadcast_to(other_frame_tokens, (batch_size, num_frames - 1, *token_tensor.shape[2:]))
        tokens = jnp.concatenate([first_frame_token, other_frame_tokens], axis=1)

    return jnp.reshape(tokens, (batch_size * num_frames, *token_tensor.shape[2:]))


class Aggregator(nn.Module):
    patch_size: int = 16
    embed_dim: int = 1024
    depth: int = 24
    num_heads: int = 16
    mlp_ratio: float = 4.0
    num_register_tokens: int = 16
    register_attention_block_indices: Tuple[int, ...] = (2, 6, 9, 14, 20)
    cached_layer_indices: Tuple[int, ...] = (4, 11, 17, 23)
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # Build patch embedding using DinoVisionTransformer:
        self.patch_embed = DinoVisionTransformer(
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            depth=24,
            num_heads=16,
            ffn_ratio=4.0,
            qkv_bias=True,
            layerscale_init=1.0e-5,
            norm_layer="layernormbf16",
            n_storage_tokens=4,
            dtype=self.dtype,
            name="patch_embed",
        )

        self.rope_embed = RopePositionEmbedding(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            base=100.0,
            normalize_coords="max",
            dtype=self.dtype,
            name="rope_embed",
        )

        self.frame_blocks = [
            SelfAttentionBlock(
                dim=self.embed_dim,
                num_heads=self.num_heads,
                ffn_ratio=self.mlp_ratio,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                init_values=1e-5,
                use_qk_norm=True,
                dtype=self.dtype,
                name=f"frame_blocks_{i}",
            )
            for i in range(self.depth)
        ]

        self.inter_frame_blocks = [
            SelfAttentionBlock(
                dim=self.embed_dim,
                num_heads=self.num_heads,
                ffn_ratio=self.mlp_ratio,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                init_values=1e-5,
                use_qk_norm=True,
                dtype=self.dtype,
                name=f"inter_frame_blocks_{i}",
            )
            for i in range(self.depth)
        ]

        self.camera_token = self.param(
            "camera_token", nn.initializers.normal(stddev=1e-3), (1, 2, 1, self.embed_dim)
        )
        self.register_token = self.param(
            "register_token", nn.initializers.normal(stddev=1e-3), (1, 2, self.num_register_tokens, self.embed_dim)
        )
        self.patch_token_start = 1 + self.num_register_tokens

        attention_types = ["global"] * self.depth
        for idx in self.register_attention_block_indices:
            attention_types[idx] = "register"
        self.inter_frame_attention_types = tuple(attention_types)

    def __call__(self, images: jnp.ndarray) -> Tuple[List[jnp.ndarray | None], int]:
        # Input shape: [B, T, H, W, 3] (channels_last)
        batch_size, num_frames, height, width, num_channels = images.shape
        assert num_channels == 3

        # ResNet standardization
        mean = jnp.array([0.485, 0.456, 0.406], dtype=self.dtype)
        std = jnp.array([0.229, 0.224, 0.225], dtype=self.dtype)
        images = (images - mean) / std

        images = jnp.reshape(images, (batch_size * num_frames, height, width, num_channels))

        camera_token = slice_expand_and_flatten(self.camera_token, batch_size, num_frames)
        register_token = slice_expand_and_flatten(self.register_token, batch_size, num_frames)

        patch_tokens = self.patch_embed(images)["x_norm_patchtokens"]

        tokens = jnp.concatenate([camera_token, register_token, patch_tokens], axis=1)
        _, num_tokens, embed_dim = tokens.shape

        patch_grid_size = (height // self.patch_size, width // self.patch_size)
        rope_sin, rope_cos = self.rope_embed(patch_grid_size[0], patch_grid_size[1])
        frame_rope = (rope_sin, rope_cos)

        outputs = []
        for block_idx in range(self.depth):
            # Frame blocks
            tokens = jnp.reshape(tokens, (batch_size * num_frames, num_tokens, embed_dim))
            tokens = self.frame_blocks[block_idx](tokens, rope=frame_rope)
            tokens.block_until_ready()
            trim_memory()
            frame_tokens = jnp.reshape(tokens, (batch_size, num_frames, num_tokens, embed_dim))

            # Inter-frame blocks
            tokens = jnp.reshape(tokens, (batch_size, num_frames, num_tokens, embed_dim))
            attention_type = self.inter_frame_attention_types[block_idx]

            if attention_type == "global":
                tokens = jnp.reshape(tokens, (batch_size, num_frames * num_tokens, embed_dim))
                tokens = self.inter_frame_blocks[block_idx](tokens, rope=None)
                tokens.block_until_ready()
                tokens = jnp.reshape(tokens, (batch_size, num_frames, num_tokens, embed_dim))
            elif attention_type == "register":
                camera_and_register_tokens = tokens[:, :, : self.patch_token_start]
                camera_and_register_tokens = jnp.reshape(
                    camera_and_register_tokens, (batch_size, num_frames * self.patch_token_start, embed_dim)
                )

                patch_tokens = tokens[:, :, self.patch_token_start:]
                patch_tokens = jnp.reshape(
                    patch_tokens, (batch_size, num_frames * (num_tokens - self.patch_token_start), embed_dim)
                )

                camera_and_register_tokens = self.inter_frame_blocks[block_idx](camera_and_register_tokens, rope=None)
                camera_and_register_tokens.block_until_ready()
                tokens = jnp.concatenate([camera_and_register_tokens, patch_tokens], axis=1)

                camera_and_register_tokens = tokens[:, : num_frames * self.patch_token_start]
                camera_and_register_tokens = jnp.reshape(
                    camera_and_register_tokens, (batch_size, num_frames, self.patch_token_start, embed_dim)
                )

                patch_tokens = tokens[:, num_frames * self.patch_token_start :]
                patch_tokens = jnp.reshape(
                    patch_tokens, (batch_size, num_frames, num_tokens - self.patch_token_start, embed_dim)
                )
                tokens = jnp.concatenate([camera_and_register_tokens, patch_tokens], axis=2)
                tokens.block_until_ready()

            trim_memory()

            if block_idx in self.cached_layer_indices:
                outputs.append(jnp.concatenate([frame_tokens, tokens], axis=-1))
            else:
                outputs.append(None)

        return outputs, self.patch_token_start


class CameraHead(nn.Module):
    dim_in: int = 2048
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.token_norm = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype, name="token_norm")
        self.trunk = [
            SelfAttentionBlock(
                dim=self.dim_in,
                num_heads=16,
                ffn_ratio=4.0,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                init_values=1e-5,
                use_qk_norm=False,
                dtype=self.dtype,
                name=f"trunk_{i}",
            )
            for i in range(4)
        ]
        self.trunk_norm = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype, name="trunk_norm")
        self.camera_branch_fc1 = nn.Dense(features=self.dim_in // 2, use_bias=True, dtype=self.dtype, name="camera_branch_0")
        self.camera_branch_fc2 = nn.Dense(features=9, use_bias=True, dtype=self.dtype, name="camera_branch_2")

    def __call__(self, aggregated_tokens_list: List[jnp.ndarray | None], patch_token_start: int) -> jnp.ndarray:
        tokens = aggregated_tokens_list[-1]
        batch_size, num_frames, num_tokens, _ = tokens.shape

        camera_and_register_tokens = tokens[:, :, :patch_token_start]
        camera_and_register_tokens = self.token_norm(camera_and_register_tokens)

        camera_and_register_tokens = jnp.reshape(camera_and_register_tokens, (batch_size, num_frames * patch_token_start, -1))
        for block in self.trunk:
            camera_and_register_tokens = block(camera_and_register_tokens, None)

        camera_and_register_tokens = jnp.reshape(camera_and_register_tokens, (batch_size, num_frames, patch_token_start, -1))
        camera_tokens = self.trunk_norm(camera_and_register_tokens[:, :, 0])

        x = self.camera_branch_fc1(camera_tokens)
        x = jax.nn.gelu(x, approximate=False)
        raw_camera = self.camera_branch_fc2(x)

        translation = raw_camera[..., :3]
        quaternion = raw_camera[..., 3:7]
        fov = jax.nn.relu(raw_camera[..., 7:]) + 0.01
        return jnp.concatenate([translation, quaternion, fov], axis=-1)


class TextAlignmentHead(nn.Module):
    dim_in: int = 2048
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.token_norm = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype, name="token_norm")
        self.language_token = self.param("language_token", nn.initializers.normal(stddev=0.02), (1, 1, self.dim_in))
        self.readout_blocks = [
            SelfAttentionBlock(
                dim=self.dim_in,
                num_heads=16,
                ffn_ratio=4.0,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                init_values=1e-5,
                use_qk_norm=False,
                dtype=self.dtype,
                name=f"readout_blocks_{i}",
            )
            for i in range(4)
        ]
        self.language_token_norm = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype, name="language_token_norm")
        self.proj_fc1 = nn.Dense(features=self.dim_in // 2, use_bias=True, dtype=self.dtype, name="embedding_projector_0")
        self.proj_ln = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype, name="embedding_projector_2")
        self.proj_fc2 = nn.Dense(features=self.dim_in, use_bias=True, dtype=self.dtype, name="embedding_projector_3")

    def __call__(self, aggregated_tokens_list: List[jnp.ndarray | None], patch_token_start: int) -> Dict[str, jnp.ndarray]:
        tokens = aggregated_tokens_list[-1]
        batch_size, num_frames, _, _ = tokens.shape

        camera_and_register_tokens = tokens[:, :, :patch_token_start]
        camera_and_register_tokens = self.token_norm(camera_and_register_tokens)
        camera_and_register_tokens = jnp.reshape(camera_and_register_tokens, (batch_size, num_frames * patch_token_start, -1))

        language_token = jnp.broadcast_to(self.language_token, (batch_size, 1, self.dim_in))
        readout_tokens = jnp.concatenate([language_token, camera_and_register_tokens], axis=1)

        for block in self.readout_blocks:
            readout_tokens = block(readout_tokens, None)

        language_token = self.language_token_norm(readout_tokens[:, 0])

        x = self.proj_fc1(language_token)
        x = jax.nn.gelu(x, approximate=False)
        x = self.proj_ln(x)
        text_alignment_embedding = self.proj_fc2(x)

        norm = jnp.linalg.norm(text_alignment_embedding, axis=-1, keepdims=True)
        norm = jnp.maximum(norm, 1e-12)
        normalized_emb = text_alignment_embedding / norm

        return {
            "text_alignment_embedding": normalized_emb,
            "text_alignment_token": language_token,
        }


def _make_dense_resize_layer(channels: int, resize_scale: float, name: str, dtype: jnp.dtype) -> Callable[[jnp.ndarray], jnp.ndarray]:
    if resize_scale == 1.0:
        return lambda x: x

    if resize_scale == 0.5:
        return nn.Conv(
            features=channels,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding=((1, 1), (1, 1)),
            use_bias=True,
            dtype=dtype,
            name=name,
        )

    upsample_scale = int(resize_scale)
    return nn.ConvTranspose(
        features=channels,
        kernel_size=(upsample_scale, upsample_scale),
        strides=(upsample_scale, upsample_scale),
        padding="VALID",
        use_bias=True,
        dtype=dtype,
        name=name,
    )


class ResidualConvUnit(nn.Module):
    features: int
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.conv1 = nn.Conv(
            features=self.features,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            use_bias=True,
            dtype=self.dtype,
            name="conv1",
        )
        self.conv2 = nn.Conv(
            features=self.features,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            use_bias=True,
            dtype=self.dtype,
            name="conv2",
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        out = jax.nn.relu(x)
        out = self.conv1(out)
        out = jax.nn.relu(out)
        out = self.conv2(out)
        return out + x


class FeatureFusionBlock(nn.Module):
    features: int
    has_residual: bool = True
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.out_conv = nn.Conv(
            features=self.features,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="VALID",
            use_bias=True,
            dtype=self.dtype,
            name="out_conv",
        )
        if self.has_residual:
            self.resConfUnit1 = ResidualConvUnit(features=self.features, dtype=self.dtype, name="resConfUnit1")
        self.resConfUnit2 = ResidualConvUnit(features=self.features, dtype=self.dtype, name="resConfUnit2")

    def __call__(self, x: jnp.ndarray, residual: jnp.ndarray | None = None, size: Tuple[int, int] | None = None) -> jnp.ndarray:
        output = x
        if self.has_residual:
            if residual is None:
                raise ValueError("FeatureFusionBlock requires a residual tensor when has_residual=True")
            output = output + self.resConfUnit1(residual)

        output = self.resConfUnit2(output)
        if size is not None:
            output = bilinear_interpolate(output, size)
        return self.out_conv(output)


class DenseHead(nn.Module):
    dim_in: int = 2048
    patch_size: int = 16
    features: int = 256
    out_channels: Tuple[int, ...] = (256, 512, 1024, 1024)
    intermediate_layer_idx: Tuple[int, ...] = (4, 11, 17, 23)
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if self.patch_size % 4 != 0:
            raise ValueError(f"DenseHead expects patch_size divisible by 4. Got patch_size={self.patch_size}.")

        self.final_shuffle_factor = self.patch_size // 4
        self.norm = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype, name="norm")

        self.projects = [
            nn.Conv(
                features=oc,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding="VALID",
                use_bias=True,
                dtype=self.dtype,
                name=f"projects_{i}",
            )
            for i, oc in enumerate(self.out_channels)
        ]

        self.resize_layers = [
            _make_dense_resize_layer(channels=self.out_channels[0], resize_scale=4.0, name="resize_layers_0", dtype=self.dtype),
            _make_dense_resize_layer(channels=self.out_channels[1], resize_scale=2.0, name="resize_layers_1", dtype=self.dtype),
            _make_dense_resize_layer(channels=self.out_channels[2], resize_scale=1.0, name="resize_layers_2", dtype=self.dtype),
            _make_dense_resize_layer(channels=self.out_channels[3], resize_scale=0.5, name="resize_layers_3", dtype=self.dtype),
        ]

        self.scratch_layer1_rn = nn.Conv(
            features=self.features,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            use_bias=False,
            dtype=self.dtype,
            name="scratch_layer1_rn",
        )
        self.scratch_layer2_rn = nn.Conv(
            features=self.features,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            use_bias=False,
            dtype=self.dtype,
            name="scratch_layer2_rn",
        )
        self.scratch_layer3_rn = nn.Conv(
            features=self.features,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            use_bias=False,
            dtype=self.dtype,
            name="scratch_layer3_rn",
        )
        self.scratch_layer4_rn = nn.Conv(
            features=self.features,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            use_bias=False,
            dtype=self.dtype,
            name="scratch_layer4_rn",
        )

        self.scratch_refinenet1 = FeatureFusionBlock(features=self.features, has_residual=True, dtype=self.dtype, name="scratch_refinenet1")
        self.scratch_refinenet2 = FeatureFusionBlock(features=self.features, has_residual=True, dtype=self.dtype, name="scratch_refinenet2")
        self.scratch_refinenet3 = FeatureFusionBlock(features=self.features, has_residual=True, dtype=self.dtype, name="scratch_refinenet3")
        self.scratch_refinenet4 = FeatureFusionBlock(features=self.features, has_residual=False, dtype=self.dtype, name="scratch_refinenet4")

        self.proj = nn.Conv(
            features=self.final_shuffle_factor**2,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="VALID",
            use_bias=True,
            dtype=self.dtype,
            name="proj",
        )
        self.proj_conf = nn.Conv(
            features=self.final_shuffle_factor**2,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="VALID",
            use_bias=True,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.constant(math.log(1.05 - 1.0)),
            dtype=self.dtype,
            name="proj_conf",
        )

    def _apply_pos_embed(self, x: jnp.ndarray, width: int, height: int, ratio: float = 0.1) -> jnp.ndarray:
        patch_w = x.shape[2]
        patch_h = x.shape[1]
        C = x.shape[3]
        pos_embed = create_uv_grid(patch_w, patch_h, aspect_ratio=width / height, dtype=x.dtype)
        pos_embed = position_grid_to_embed(pos_embed, C)
        pos_embed = pos_embed * ratio
        return x + pos_embed[None, :, :, :]

    def scratch_forward(self, features: List[jnp.ndarray]) -> jnp.ndarray:
        layer_1, layer_2, layer_3, layer_4 = features

        layer_1_rn = self.scratch_layer1_rn(layer_1)
        layer_2_rn = self.scratch_layer2_rn(layer_2)
        layer_3_rn = self.scratch_layer3_rn(layer_3)
        layer_4_rn = self.scratch_layer4_rn(layer_4)

        out = self.scratch_refinenet4(layer_4_rn, size=layer_3_rn.shape[1:3])
        out = self.scratch_refinenet3(out, layer_3_rn, size=layer_2_rn.shape[1:3])
        out = self.scratch_refinenet2(out, layer_2_rn, size=layer_1_rn.shape[1:3])
        return self.scratch_refinenet1(out, layer_1_rn, size=layer_1_rn.shape[1:3])

    def __call__(
        self,
        aggregated_tokens_list: List[jnp.ndarray | None],
        images: jnp.ndarray,
        patch_token_start: int,
        frames_chunk_size: int | None = 8,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Input shape: [B, T, H, W, 3]
        batch_size, num_frames, height, width, _ = images.shape
        patch_h, patch_w = height // self.patch_size, width // self.patch_size

        # In JAX execution, frames_chunk_size is not strictly needed for speed since VLAX handles it,
        # but to keep exact logical parity, we can implement the forward implementation directly.
        multi_scale_features = []
        for feature_idx, layer_idx in enumerate(self.intermediate_layer_idx):
            x = aggregated_tokens_list[layer_idx]
            if x is None:
                raise ValueError(f"Aggregator did not cache layer {layer_idx}, which DenseHead needs.")
            x = x[:, :, patch_token_start:]
            x = x.astype(jnp.float32)

            x = jnp.reshape(x, (batch_size * num_frames, -1, x.shape[-1]))
            x = self.norm(x)
            
            # Reshape to NHWC:
            x = jnp.reshape(x, (x.shape[0], patch_h, patch_w, x.shape[-1]))
            x = self.projects[feature_idx](x)
            x = self._apply_pos_embed(x, width, height)
            x = self.resize_layers[feature_idx](x)
            multi_scale_features.append(x)

        fused = self.scratch_forward(multi_scale_features)
        fused = self._apply_pos_embed(fused, width, height)

        depth_logits = self.proj(fused)
        depth_logits = pixel_shuffle(depth_logits, self.final_shuffle_factor)

        confidence_logits = self.proj_conf(fused)
        confidence_logits = pixel_shuffle(confidence_logits, self.final_shuffle_factor)
        confidence_logits = jnp.squeeze(confidence_logits, axis=-1)

        depth = jnp.exp(depth_logits)
        depth_conf = 1.0 + jnp.exp(confidence_logits)

        depth = jnp.reshape(depth, (batch_size, num_frames, *depth.shape[1:]))
        depth_conf = jnp.reshape(depth_conf, (batch_size, num_frames, *depth_conf.shape[1:]))

        return depth, depth_conf


class VGGTOmega(nn.Module):
    patch_size: int = 16
    embed_dim: int = 1024
    enable_camera: bool = True
    enable_depth: bool = True
    enable_alignment: bool = False
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.aggregator = Aggregator(patch_size=self.patch_size, embed_dim=self.embed_dim, dtype=self.dtype, name="aggregator")

        if self.enable_camera:
            self.camera_head = CameraHead(dim_in=2 * self.embed_dim, dtype=self.dtype, name="camera_head")
        else:
            self.camera_head = None

        if self.enable_depth:
            self.dense_head = DenseHead(dim_in=2 * self.embed_dim, patch_size=self.patch_size, dtype=self.dtype, name="dense_head")
        else:
            self.dense_head = None

        if self.enable_alignment:
            self.text_alignment_head = TextAlignmentHead(dim_in=2 * self.embed_dim, dtype=self.dtype, name="text_alignment_head")
        else:
            self.text_alignment_head = None

    def __call__(self, images: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        # Input images shape: [B, T, H, W, 3]
        if len(images.shape) == 4:
            images = images[None, :, :, :, :]

        batch_size, num_frames, height, width, num_channels = images.shape

        # Run aggregator
        aggregated_tokens_list, patch_token_start = self.aggregator(images)

        final_tokens = aggregated_tokens_list[-1]
        if final_tokens is None:
            raise ValueError("Aggregator did not cache the final layer, which VGGTOmega needs.")

        predictions = {
            "camera_and_register_tokens": final_tokens[:, :, :patch_token_start],
        }

        if self.camera_head is not None:
            predictions["pose_enc"] = self.camera_head(aggregated_tokens_list, patch_token_start)

        if self.dense_head is not None:
            depth, depth_conf = self.dense_head(aggregated_tokens_list, images, patch_token_start)
            predictions["depth"] = depth
            predictions["depth_conf"] = depth_conf

        if self.text_alignment_head is not None:
            predictions.update(self.text_alignment_head(aggregated_tokens_list, patch_token_start))

        return predictions

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Callable, Tuple, Any
import jax
import jax.numpy as jnp
import flax.linen as nn


class RMSNorm(nn.Module):
    dim: int
    eps: float = 1e-5
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        weight = self.param("weight", nn.initializers.ones, (self.dim,))
        # Cast to float32 for computation parity
        x_f32 = x.astype(jnp.float32)
        variance = jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True)
        norm_x = x_f32 * jax.lax.rsqrt(variance + self.eps)
        norm_x = norm_x.astype(self.dtype)
        return norm_x * weight.astype(self.dtype)


class LayerScale(nn.Module):
    dim: int
    init_values: float = 1e-5
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        gamma = self.param("gamma", nn.initializers.constant(self.init_values), (self.dim,))
        return x * gamma.astype(self.dtype)


class PatchEmbed(nn.Module):
    patch_size: int = 16
    embed_dim: int = 768
    flatten_embedding: bool = True
    norm_layer: Callable | None = None
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Input shape: [B, H, W, C]
        # In JAX/Flax nn.Conv kernel_size is a tuple, default feature dimension is channels_last
        # Weight shape in nn.Conv: [kernel_h, kernel_w, in_channels, out_channels]
        x = nn.Conv(
            features=self.embed_dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding="VALID",
            use_bias=True,
            dtype=self.dtype,
            name="proj",
        )(x)
        
        # Output shape: [B, H_patch, W_patch, embed_dim]
        H, W = x.shape[1], x.shape[2]
        if self.flatten_embedding:
            x = jnp.reshape(x, (x.shape[0], -1, self.embed_dim))
            
        if self.norm_layer is not None:
            x = self.norm_layer(name="norm")(x)
            
        return x


class RopePositionEmbedding(nn.Module):
    embed_dim: int
    num_heads: int
    base: float | None = 100.0
    min_period: float | None = None
    max_period: float | None = None
    normalize_coords: str = "separate"
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        D_head = self.embed_dim // self.num_heads
        assert self.embed_dim % (4 * self.num_heads) == 0
        
        def init_periods(key):
            if self.base is not None:
                periods = self.base ** (
                    2.0 * jnp.arange(D_head // 4, dtype=jnp.float32) / (D_head // 2)
                )
            else:
                base = self.max_period / self.min_period
                exponents = jnp.linspace(0.0, 1.0, D_head // 4, dtype=jnp.float32)
                periods = base**exponents
                periods = periods / base
                periods = periods * self.max_period
            return periods.astype(self.dtype)
            
        self.periods = self.param("periods", init_periods)

    def __call__(self, H: int, W: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Prepare coords in range [-1, +1]
        if self.normalize_coords == "max":
            max_HW = max(H, W)
            coords_h = (jnp.arange(H, dtype=self.dtype) + 0.5) / max_HW
            coords_w = (jnp.arange(W, dtype=self.dtype) + 0.5) / max_HW
        elif self.normalize_coords == "min":
            min_HW = min(H, W)
            coords_h = (jnp.arange(H, dtype=self.dtype) + 0.5) / min_HW
            coords_w = (jnp.arange(W, dtype=self.dtype) + 0.5) / min_HW
        elif self.normalize_coords == "separate":
            coords_h = (jnp.arange(H, dtype=self.dtype) + 0.5) / H
            coords_w = (jnp.arange(W, dtype=self.dtype) + 0.5) / W
        else:
            raise ValueError(f"Unknown normalize_coords: {self.normalize_coords}")

        grid_h, grid_w = jnp.meshgrid(coords_h, coords_w, indexing="ij")
        coords = jnp.stack([grid_h, grid_w], axis=-1)  # [H, W, 2]
        coords = jnp.reshape(coords, (-1, 2))  # [HW, 2]
        coords = 2.0 * coords - 1.0

        # Prepare angles and sin/cos
        angles = 2.0 * jnp.pi * coords[:, :, None] / self.periods[None, None, :]  # [HW, 2, D_head // 4]
        angles = jnp.reshape(angles, (angles.shape[0], -1))  # [HW, D_head // 2]
        angles = jnp.tile(angles, (1, 2))  # [HW, D_head]
        cos = jnp.cos(angles)
        sin = jnp.sin(angles)

        return sin, cos


class SelfAttention(nn.Module):
    dim: int
    num_heads: int = 8
    qkv_bias: bool = False
    proj_bias: bool = True
    use_qk_norm: bool = False
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.head_dim = self.dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Dense(features=self.dim * 3, use_bias=self.qkv_bias, dtype=self.dtype, name="qkv")
        self.proj = nn.Dense(features=self.dim, use_bias=self.proj_bias, dtype=self.dtype, name="proj")
        if self.use_qk_norm:
            self.q_norm = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype, name="q_norm")
            self.k_norm = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype, name="k_norm")

    def rope_rotate_half(self, x: jnp.ndarray) -> jnp.ndarray:
        d = x.shape[-1]
        x1 = x[..., : d // 2]
        x2 = x[..., d // 2 :]
        return jnp.concatenate([-x2, x1], axis=-1)

    def rope_apply(self, x: jnp.ndarray, sin: jnp.ndarray, cos: jnp.ndarray) -> jnp.ndarray:
        return (x * cos) + (self.rope_rotate_half(x) * sin)

    def apply_rope(self, q: jnp.ndarray, k: jnp.ndarray, rope: Tuple[jnp.ndarray, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray]:
        sin, cos = rope
        sin = sin.astype(q.dtype)
        cos = cos.astype(q.dtype)
        N = q.shape[-2]
        prefix = N - sin.shape[-2]

        q_prefix = q[:, :, :prefix, :]
        q_suffix = self.rope_apply(q[:, :, prefix:, :], sin[None, None, :, :], cos[None, None, :, :])
        q = jnp.concatenate([q_prefix, q_suffix], axis=-2)

        k_prefix = k[:, :, :prefix, :]
        k_suffix = self.rope_apply(k[:, :, prefix:, :], sin[None, None, :, :], cos[None, None, :, :])
        k = jnp.concatenate([k_prefix, k_suffix], axis=-2)

        return q, k

    def __call__(self, x: jnp.ndarray, rope: Tuple[jnp.ndarray, jnp.ndarray] | None = None) -> jnp.ndarray:
        B, N, C = x.shape
        qkv = self.qkv(x)  # [B, N, 3 * C]
        qkv = jnp.reshape(qkv, (B, N, 3, self.num_heads, self.head_dim))
        q = qkv[:, :, 0, :, :]
        k = qkv[:, :, 1, :, :]
        v = qkv[:, :, 2, :, :]

        q = jnp.transpose(q, (0, 2, 1, 3))  # [B, num_heads, N, head_dim]
        k = jnp.transpose(k, (0, 2, 1, 3))  # [B, num_heads, N, head_dim]
        v = jnp.transpose(v, (0, 2, 1, 3))  # [B, num_heads, N, head_dim]

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if rope is not None:
            q, k = self.apply_rope(q, k, rope)

        # Standard self-attention
        attn_weights = jnp.matmul(q, jnp.swapaxes(k, -1, -2)) * self.scale
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        attn_out = jnp.matmul(attn_weights, v)

        attn_out = jnp.transpose(attn_out, (0, 2, 1, 3))
        attn_out = jnp.reshape(attn_out, (B, N, C))

        return self.proj(attn_out)


class Mlp(nn.Module):
    in_features: int
    hidden_features: int | None = None
    out_features: int | None = None
    bias: bool = True
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        hidden = self.hidden_features or self.in_features
        out = self.out_features or self.in_features
        self.fc1 = nn.Dense(features=hidden, use_bias=self.bias, dtype=self.dtype, name="fc1")
        self.fc2 = nn.Dense(features=out, use_bias=self.bias, dtype=self.dtype, name="fc2")

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.fc1(x)
        x = jax.nn.gelu(x, approximate=False)
        x = self.fc2(x)
        return x


@nn.jit
class SelfAttentionBlock(nn.Module):
    dim: int
    num_heads: int
    ffn_ratio: float = 4.0
    qkv_bias: bool = False
    proj_bias: bool = True
    ffn_bias: bool = True
    init_values: float | None = None
    use_qk_norm: bool = False
    norm_layer: str = "layernorm"
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if self.norm_layer == "rmsnorm":
            self.norm1 = RMSNorm(dim=self.dim, dtype=self.dtype, name="norm1")
            self.norm2 = RMSNorm(dim=self.dim, dtype=self.dtype, name="norm2")
        else:
            self.norm1 = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype, name="norm1")
            self.norm2 = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype, name="norm2")

        self.attn = SelfAttention(
            dim=self.dim,
            num_heads=self.num_heads,
            qkv_bias=self.qkv_bias,
            proj_bias=self.proj_bias,
            use_qk_norm=self.use_qk_norm,
            dtype=self.dtype,
            name="attn",
        )

        if self.init_values is not None:
            self.ls1 = LayerScale(dim=self.dim, init_values=self.init_values, dtype=self.dtype, name="ls1")
            self.ls2 = LayerScale(dim=self.dim, init_values=self.init_values, dtype=self.dtype, name="ls2")
        else:
            self.ls1 = None
            self.ls2 = None

        self.mlp = Mlp(
            in_features=self.dim,
            hidden_features=int(self.dim * self.ffn_ratio),
            bias=self.ffn_bias,
            dtype=self.dtype,
            name="mlp",
        )

    def __call__(self, x: jnp.ndarray, rope: Tuple[jnp.ndarray, jnp.ndarray] | None = None) -> jnp.ndarray:
        residual = self.attn(self.norm1(x), rope=rope)
        if self.ls1 is not None:
            residual = self.ls1(residual)
        x_attn = x + residual

        residual_mlp = self.mlp(self.norm2(x_attn))
        if self.ls2 is not None:
            residual_mlp = self.ls2(residual_mlp)
        x_ffn = x_attn + residual_mlp

        return x_ffn

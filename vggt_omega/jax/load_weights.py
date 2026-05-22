# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import zstandard as zstd
import flax.serialization

def load_checkpoint(params_template, filepath: str):
    """
    Decompresses and deserializes JAX/Flax weights from a zstd-compressed msgpack file.
    
    Args:
        params_template: A nested dictionary/template structure (e.g. from model.init)
                        that matches the shape/keys of the saved parameters.
        filepath: Path to the .msgpack.zst file.
        
    Returns:
        The restored parameter dictionary.
    """
    with open(filepath, "rb") as f:
        compressed_bytes = f.read()
        
    dctx = zstd.ZstdDecompressor()
    serialized_bytes = dctx.decompress(compressed_bytes)
    
    # Restore the structure matching the template
    restored = flax.serialization.from_bytes(params_template, serialized_bytes)
    return restored

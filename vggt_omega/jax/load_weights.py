import zstandard as zstd
import flax.serialization
import gc

def load_checkpoint(params_template, filepath: str):
    """
    Decompresses and deserializes JAX/Flax weights from a zstd-compressed msgpack file.
    
    Args:
        params_template: A nested dictionary/template structure (e.g. from model.init)
                        that matches the shape/keys of the saved parameters, or None
                        for direct template-free dictionary restoration.
        filepath: Path to the .msgpack.zst file.
        
    Returns:
        The restored parameter dictionary.
    """
    with open(filepath, "rb") as f:
        compressed_bytes = f.read()
        
    dctx = zstd.ZstdDecompressor()
    serialized_bytes = dctx.decompress(compressed_bytes)
    
    # Free compressed memory
    del compressed_bytes
    gc.collect()
    
    if params_template is None:
        restored = flax.serialization.msgpack_restore(serialized_bytes)
    else:
        # Restore the structure matching the template
        restored = flax.serialization.from_bytes(params_template, serialized_bytes)
        
    # Free serialized bytes memory
    del serialized_bytes
    gc.collect()
    
    return restored

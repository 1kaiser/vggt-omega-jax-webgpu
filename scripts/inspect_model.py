import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
import jax
import jax.numpy as jnp
from vggt_omega.models import VGGTOmega as PTModel
from vggt_omega.jax.models import VGGTOmega as JAXModel

# Create models
pt_model = PTModel(enable_camera=True, enable_depth=True, enable_alignment=True)
pt_keys = sorted(list(pt_model.state_dict().keys()))

# Flax model needs initialization to get parameter keys
jax_model = JAXModel(enable_camera=True, enable_depth=True, enable_alignment=True)
dummy_img = jnp.zeros((1, 2, 224, 224, 3))
variables = jax_model.init(jax.random.PRNGKey(0), dummy_img)
params = variables["params"]

def get_flax_keys(d, prefix=""):
    keys = []
    for k, v in d.items():
        name = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict) or hasattr(v, "keys"):
            keys.extend(get_flax_keys(v, name))
        else:
            keys.append(name)
    return keys

jax_keys = sorted(get_flax_keys(params))

print(f"PyTorch state_dict keys count: {len(pt_keys)}")
print(f"JAX/Flax parameter keys count: {len(jax_keys)}")

print("\n--- FIRST 20 PYTORCH KEYS ---")
for k in pt_keys[:20]:
    print(k)

print("\n--- FIRST 20 JAX KEYS ---")
for k in jax_keys[:20]:
    print(k)

# Save to a file for comparison
with open("pt_keys.txt", "w") as f:
    f.write("\n".join(pt_keys))
with open("jax_keys.txt", "w") as f:
    f.write("\n".join(jax_keys))
print("Saved all keys to pt_keys.txt and jax_keys.txt")

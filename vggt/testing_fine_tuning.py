from huggingface_hub import hf_hub_download
import torch

from vggt.models.vggt_modifying import VGGT

print("=" * 80)
print("Downloading Official Checkpoint")
print("=" * 80)

# Download official checkpoint
ckpt_path = hf_hub_download(
    repo_id="facebook/VGGT-1B",
    filename="model.safetensors",
)

print("Checkpoint:", ckpt_path)

# Load checkpoint
checkpoint = torch.load(ckpt_path, map_location="cpu")
checkpoint_keys = set(checkpoint.keys())

print("\nCheckpoint tensors:", len(checkpoint_keys))

print("\n" + "=" * 80)
print("Building Modified Model")
print("=" * 80)

model = VGGT()
model_keys = set(model.state_dict().keys())

print("Model tensors:", len(model_keys))

# ------------------------------------------------------------------
# Compare
# ------------------------------------------------------------------

missing = sorted(model_keys - checkpoint_keys)
unexpected = sorted(checkpoint_keys - model_keys)
common = sorted(model_keys & checkpoint_keys)

print("\n" + "=" * 80)
print("Summary")
print("=" * 80)

print(f"Common Keys      : {len(common)}")
print(f"Missing Keys     : {len(missing)}")
print(f"Unexpected Keys  : {len(unexpected)}")

# ------------------------------------------------------------------
# Missing
# ------------------------------------------------------------------

print("\n" + "=" * 80)
print("Missing Keys")
print("=" * 80)

if len(missing) == 0:
    print("None")
else:
    for k in missing:
        print(k)

# ------------------------------------------------------------------
# Unexpected
# ------------------------------------------------------------------

print("\n" + "=" * 80)
print("Unexpected Keys")
print("=" * 80)

if len(unexpected) == 0:
    print("None")
else:
    for k in unexpected:
        print(k)

# ------------------------------------------------------------------
# Shape mismatches
# ------------------------------------------------------------------

print("\n" + "=" * 80)
print("Checking Shapes")
print("=" * 80)

shape_errors = 0

model_state = model.state_dict()

for k in common:

    if checkpoint[k].shape != model_state[k].shape:

        shape_errors += 1

        print(
            f"{k}\n"
            f"Checkpoint: {tuple(checkpoint[k].shape)}\n"
            f"Model     : {tuple(model_state[k].shape)}\n"
        )

if shape_errors == 0:
    print("✓ All common tensors have matching shapes.")

print("\nVerification Complete!")
import torch
from vggt.models.vggt_modifying import VGGT

model = VGGT()
model.eval()

images = torch.randn(1, 1, 3, 518, 518)

with torch.no_grad():
    outputs = model(images)

print(outputs.keys())

for k, v in outputs.items():
    if isinstance(v, torch.Tensor):
        print(f"{k:15s} {tuple(v.shape)}")
    elif isinstance(v, list):
        print(f"{k:15s} List of length {len(v)}")
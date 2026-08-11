import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from fine_tuning.model_builder import build_model

model = build_model()

checkpoint = torch.load(
    "checkpoints/best_model.pth",
    map_location="cpu"
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()
model.cuda()

x = torch.randn(
    1,
    1,
    3,
    518,
    518,
    device="cuda"
)

with torch.no_grad():
    outputs = model(x)

print(outputs.keys())

for k, v in outputs.items():
    if torch.is_tensor(v):
        print(k, v.shape)


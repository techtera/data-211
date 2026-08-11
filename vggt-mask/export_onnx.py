import torch

from fine_tuning.model_builder import build_model

CHECKPOINT_PATH = "checkpoints/best_model.pth"

print("Loading model...")

model = build_model()

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu"
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

print("Moving model to CPU...")

model = model.cpu()

if hasattr(model.aggregator, "_resnet_mean"):
    model.aggregator._resnet_mean = (
        model.aggregator._resnet_mean.cpu()
    )

if hasattr(model.aggregator, "_resnet_std"):
    model.aggregator._resnet_std = (
        model.aggregator._resnet_std.cpu()
    )

print("Model device:",
      next(model.parameters()).device)

print("Mean device:",
      model.aggregator._resnet_mean.device)

print("Std device:",
      model.aggregator._resnet_std.device)

model.eval()

dummy_input = torch.randn(
    1,
    1,
    3,
    518,
    518,
)

print("Exporting ONNX...")

torch.onnx.export(
    model,
    dummy_input,
    "vggt_segformer.onnx",
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["mask_logits"],
)

print("ONNX export complete.")

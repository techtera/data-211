import torch
import matplotlib.pyplot as plt

from vggt.models.vggt_modifying import VGGT

############################################################
# Create Model
############################################################

device = "cuda" if torch.cuda.is_available() else "cpu"

model = VGGT().to(device)

print("=" * 60)
print("Model Loaded Successfully")
print("=" * 60)

############################################################
# Dummy Input
############################################################
image_path = "/Users/dikshitrishi/Terafac/images/1785389716853308652_1.png"

from vggt.utils.load_fn import load_and_preprocess_images
images = load_and_preprocess_images([image_path]).to(device)


# images = torch.randn(
#     1,      # Batch
#     1,      # Frames
#     3,
#     518,
#     518,
# ).to(device)

############################################################
# Forward Pass
############################################################

model.eval()

with torch.no_grad():
    predictions = model(images)

############################################################
# Prediction Keys
############################################################

print("\nPrediction Keys")
print("-" * 60)
print(predictions.keys())

############################################################
# Prediction Shapes
############################################################

print("\nPrediction Shapes")
print("-" * 60)

for k, v in predictions.items():

    if isinstance(v, torch.Tensor):
        print(f"{k:15s} {tuple(v.shape)}")

    elif isinstance(v, list):
        print(f"{k:15s} List of length {len(v)}")

############################################################
# Verify Segmentation Output
############################################################

mask_logits = predictions["mask_logits"]

print("\nSegmentation Output")
print("-" * 60)
print("Shape :", mask_logits.shape)
print("Min   :", mask_logits.min().item())
print("Max   :", mask_logits.max().item())
print("Mean  :", mask_logits.mean().item())

############################################################
# Convert logits -> predicted mask
############################################################

pred_mask = mask_logits[0, 0].argmax(dim=0).cpu()

print("\nUnique Predicted Classes")
print("-" * 60)
print(torch.unique(pred_mask))

############################################################
# Visualize
############################################################

plt.figure(figsize=(6,6))
plt.imshow(pred_mask)
plt.title("Predicted Segmentation Mask")
plt.colorbar()
plt.axis("off")
plt.show()

# ############################################################
# # Gradient Check
# ############################################################

# print("\nRunning Gradient Check...")
# print("-" * 60)

# model.train()

# images = torch.randn(
#     1,
#     1,
#     3,
#     518,
#     518,
# ).to(device)

# predictions = model(images)

# loss = predictions["mask_logits"].mean()

# loss.backward()

# decoder_grad = False

# for name, param in model.named_parameters():

#     if "segformer_decoder" in name:

#         if param.grad is not None:

#             decoder_grad = True
#             print(f"Gradient OK : {name}")

# print()

# if decoder_grad:
#     print("✓ SegFormer Decoder receives gradients.")
# else:
#     print("✗ No gradients found!")

# print("\nVerification Complete!")
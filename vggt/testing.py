import torch

from vggt.models.vggt_modifying import VGGT

random_model = VGGT()
pretrained_model = VGGT.from_pretrained("facebook/VGGT-1B")

for name, param in pretrained_model.aggregator.state_dict().items():

    if not torch.equal(
        param,
        random_model.aggregator.state_dict()[name]
    ):
        print(f"Different: {name}")
        break
from torch.utils.data import DataLoader, Subset

from .config import (
    DATASET_ROOT,
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
)

from .dataset import SegmentationDataset


def build_dataloader():

    dataset = SegmentationDataset(
        root_dir=DATASET_ROOT,
        image_size=IMAGE_SIZE,
    )
    dataset = Subset(dataset, range(20))
    
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return dataloader
"""
TensorBoard Logger.
"""

from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from .config import LOG_DIR


# ============================================================
# Build TensorBoard Writer
# ============================================================

def build_writer():
    """
    Create a TensorBoard SummaryWriter.
    """

    Path(LOG_DIR).mkdir(
        parents=True,
        exist_ok=True,
    )

    writer = SummaryWriter(LOG_DIR)

    print("=" * 60)
    print("TensorBoard Logger")
    print("=" * 60)
    print(f"✓ Log Directory : {LOG_DIR}")

    return writer
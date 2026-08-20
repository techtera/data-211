"""
Apply VGGT block truncation to model files.

Usage:
    # Truncate to 20 blocks
    python apply_truncation.py --depth 20

    # Truncate to 18 blocks
    python apply_truncation.py --depth 18

    # Restore original files
    python apply_truncation.py --restore

Modifies:
    - encoder/aggregator.py (depth, cached_layer_indices)
    - decoders/obj_mask/segformer_head.py (intermediate_layer_idx)

Creates backups:
    - encoder/aggregator.py.backup
    - decoders/obj_mask/segformer_head.py.backup
"""

import argparse
import shutil
from pathlib import Path


DEPTH_CONFIGS = {
    24: [4, 11, 17, 23],  # Original
    22: [4, 11, 17, 21],
    20: [4, 11, 16, 19],
    18: [4, 10, 14, 17],
    16: [3, 8, 12, 15],
}


def backup_file(filepath):
    """Create backup of file."""
    backup_path = Path(str(filepath) + ".backup")
    if not backup_path.exists():
        shutil.copy2(filepath, backup_path)
        print(f"✅ Backed up: {filepath}")
    else:
        print(f"ℹ️  Backup exists: {backup_path}")


def restore_file(filepath):
    """Restore file from backup."""
    backup_path = Path(str(filepath) + ".backup")
    if backup_path.exists():
        shutil.copy2(backup_path, filepath)
        print(f"✅ Restored: {filepath}")
        return True
    else:
        print(f"⚠️  No backup found: {backup_path}")
        return False


def modify_aggregator(depth, cached_layers):
    """Modify encoder/aggregator.py."""
    filepath = Path("encoder/aggregator.py")

    if not filepath.exists():
        print(f"❌ File not found: {filepath}")
        return False

    backup_file(filepath)

    with open(filepath, 'r') as f:
        lines = f.readlines()

    modified = False

    for i, line in enumerate(lines):
        # Look for depth parameter in __init__
        if 'depth=' in line and i < 100:  # Should be near top
            # Check if it's one of the known depth values
            for d in [24, 22, 20, 18, 16]:
                if f'depth={d}' in line:
                    old_line = line
                    lines[i] = line.replace(f'depth={d}', f'depth={depth}')
                    if lines[i] != old_line:
                        print(f"✏️  Line {i+1}: depth={depth}")
                        modified = True
                    break

        # Look for cached_layer_indices
        if 'cached_layer_indices' in line and '=' in line:
            # Check if it contains a tuple with layer indices
            if '(' in line and ')' in line:
                old_line = line
                indent = len(line) - len(line.lstrip())
                new_layers_str = ', '.join(map(str, cached_layers))
                # Reconstruct the line
                if 'cached_layer_indices:' in line:
                    # Type hint version
                    lines[i] = ' ' * indent + f"cached_layer_indices: Tuple[int, ...] = ({new_layers_str}),\n"
                else:
                    lines[i] = ' ' * indent + f"cached_layer_indices=({new_layers_str}),\n"
                if lines[i] != old_line:
                    print(f"✏️  Line {i+1}: cached_layer_indices={cached_layers}")
                    modified = True

    if modified:
        with open(filepath, 'w') as f:
            f.writelines(lines)
        print(f"✅ Modified: {filepath}")
        return True
    else:
        print(f"⚠️  No changes in: {filepath}")
        return False


def modify_segformer_head(cached_layers):
    """Modify decoders/obj_mask/segformer_head.py."""
    filepath = Path("decoders/obj_mask/segformer_head.py")

    if not filepath.exists():
        print(f"❌ File not found: {filepath}")
        return False

    backup_file(filepath)

    with open(filepath, 'r') as f:
        lines = f.readlines()

    modified = False

    for i, line in enumerate(lines):
        if 'intermediate_layer_idx' in line and '=' in line:
            if '[' in line and ']' in line:
                old_line = line
                indent = len(line) - len(line.lstrip())
                new_layers_str = ', '.join(map(str, cached_layers))

                if 'intermediate_layer_idx:' in line:
                    lines[i] = ' ' * indent + f"intermediate_layer_idx: List[int] = [{new_layers_str}],\n"
                else:
                    lines[i] = ' ' * indent + f"intermediate_layer_idx=[{new_layers_str}],\n"

                if lines[i] != old_line:
                    print(f"✏️  Line {i+1}: intermediate_layer_idx={cached_layers}")
                    modified = True

    if modified:
        with open(filepath, 'w') as f:
            f.writelines(lines)
        print(f"✅ Modified: {filepath}")
        return True
    else:
        print(f"⚠️  No changes in: {filepath}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Apply VGGT truncation")
    parser.add_argument("--depth", type=int, choices=[24, 22, 20, 18, 16],
                       help="Target depth (number of block pairs)")
    parser.add_argument("--restore", action="store_true",
                       help="Restore original files from backups")
    args = parser.parse_args()

    if args.restore:
        print("="*70)
        print("RESTORING ORIGINAL FILES")
        print("="*70)
        restored1 = restore_file("encoder/aggregator.py")
        restored2 = restore_file("decoders/obj_mask/segformer_head.py")

        if restored1 or restored2:
            print("\n✅ Restoration complete")
        else:
            print("\n⚠️  No backups found")
        return

    if args.depth is None:
        parser.print_help()
        return

    depth = args.depth
    cached_layers = DEPTH_CONFIGS[depth]

    print("="*70)
    print(f"APPLYING TRUNCATION: {depth} BLOCKS")
    print("="*70)
    print(f"Depth: {depth}")
    print(f"Cached layers: {cached_layers}")
    print()

    # Modify files
    success1 = modify_aggregator(depth, cached_layers)
    print()
    success2 = modify_segformer_head(cached_layers)

    print()
    print("="*70)
    if success1 or success2:
        print("✅ TRUNCATION APPLIED")
        print("="*70)
        print()
        print("Next steps:")
        print("  1. Run inference:")
        print(f"     python run_inference_save.py --checkpoint /checkpoints/vggt_unified_fp16.pt \\")
        print(f"         --images_dir /path/to/images --output_dir predictions/truncated_{depth}blocks")
        print()
        print("  2. Compare with baseline:")
        print(f"     python compare_visual.py --baseline_dir predictions/baseline_24blocks \\")
        print(f"         --test_dir predictions/truncated_{depth}blocks \\")
        print(f"         --images_dir /path/to/images --output_dir comparisons/24vs{depth}blocks")
        print()
        print("  3. If quality degrades, restore:")
        print("     python apply_truncation.py --restore")
    else:
        print("⚠️  NO CHANGES APPLIED")
        print("="*70)


if __name__ == "__main__":
    main()

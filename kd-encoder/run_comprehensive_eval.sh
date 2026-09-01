#!/bin/bash
#
# Run comprehensive student encoder evaluation
#
# This will measure:
# - Cross-correlation with teacher
# - Feature variance (discriminative power)
# - Activation sparsity (information density)
# - Feature statistics
#

echo "================================"
echo "Phase 1: Comprehensive Feature Evaluation"
echo "================================"
echo ""

# Configuration
STUDENT_CKPT="checkpoints_full/student_final.pt"
TEACHER_CKPT="../../vggt-unified/checkpoints/vggt_unified_fp16.pt"
TEST_IMAGES="../../rgb_reg/*.png"
MAX_IMAGES=50

# Check if checkpoints exist
if [ ! -f "$STUDENT_CKPT" ]; then
    echo "❌ Student checkpoint not found: $STUDENT_CKPT"
    echo "   Available checkpoints:"
    ls -lh checkpoints_full/*.pt 2>/dev/null || echo "   No checkpoints found!"
    exit 1
fi

if [ ! -f "$TEACHER_CKPT" ]; then
    echo "❌ Teacher checkpoint not found: $TEACHER_CKPT"
    exit 1
fi

echo "✓ Student checkpoint: $STUDENT_CKPT"
echo "✓ Teacher checkpoint: $TEACHER_CKPT"
echo "✓ Test images: $TEST_IMAGES"
echo "✓ Max images: $MAX_IMAGES"
echo ""

# Run evaluation
python evaluate_features.py \
    --student "$STUDENT_CKPT" \
    --teacher "$TEACHER_CKPT" \
    --images "$TEST_IMAGES" \
    --max_images $MAX_IMAGES \
    --device cuda

echo ""
echo "================================"
echo "Evaluation Complete"
echo "================================"

#!/bin/bash
# Comprehensive diagnostic script for VM training issues

echo "=========================================="
echo "VM TRAINING ISSUE DIAGNOSTIC"
echo "=========================================="

echo -e "\n[1] Git status and branch"
git status
git branch
git log --oneline -3

echo -e "\n[2] Check train_ddp.py StudentAggregator line"
grep -n "StudentAggregator(" train_ddp.py

echo -e "\n[3] Check aggregator.py default embed_dim"
grep -A5 "def __init__" student/aggregator.py | grep "embed_dim"

echo -e "\n[4] List Python cache directories"
find . -type d -name "__pycache__" | head -10

echo -e "\n[5] Check if old checkpoint exists"
ls -lh checkpoints_v2/ 2>/dev/null || echo "No checkpoints_v2 directory"

echo -e "\n[6] Test StudentAggregator instantiation"
python3 << 'PYEOF'
import sys
sys.path.insert(0, '.')

# Clear any cached imports
if 'student' in sys.modules:
    del sys.modules['student']
if 'student.aggregator' in sys.modules:
    del sys.modules['student.aggregator']

from student import StudentAggregator

# Test default instantiation
student = StudentAggregator()
qkv_shape = student.frame_blocks[0].attn.qkv.weight.shape
print(f"QKV shape: {qkv_shape}")
print(f"Expected: torch.Size([2304, 768])")
if qkv_shape[0] == 2304 and qkv_shape[1] == 768:
    print("✅ CORRECT - 768-dim model")
elif qkv_shape[0] == 3072 and qkv_shape[1] == 1024:
    print("❌ WRONG - 1024-dim model (OLD VERSION)")
else:
    print(f"❌ UNKNOWN - unexpected shape")
PYEOF

echo -e "\n[7] Check which student/aggregator.py is being imported"
python3 << 'PYEOF'
import sys
sys.path.insert(0, '.')
import student.aggregator
print(f"Loaded from: {student.aggregator.__file__}")
PYEOF

echo -e "\n=========================================="
echo "RECOMMENDED FIXES:"
echo "=========================================="
echo "1. Clear Python cache: find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null"
echo "2. Delete old checkpoints: rm -rf checkpoints_v2/"
echo "3. Hard reset git: git reset --hard HEAD && git pull"
echo "4. Restart Python: pkill -f python"
echo "5. Then start training fresh"

# Quick Start - Performance Testing

## 🏃 Run These Tests in Order:

### 1. Test GPU Batching Fix (Most Important!)
```powershell
py test_batch_optimization.py
```
**What it tests**: GPU transfer optimization in training
**Expected**: 2-5x faster training on GPU

### 2. Test All Optimizations Together  
```powershell
py test_all_optimizations.py
```
**What it tests**: Complete 200-step episode with all optimizations
**Expected**: 
- Baseline: 5-8s for 200 steps
- Enhanced: 8-12s for 200 steps
- Training < 20% of time

### 3. Benchmark Individual Components
```powershell
py benchmark_optimizations.py
```
**What it tests**: Ray-casting and encoding speeds
**Expected**:
- Ray-casting: <5ms per call
- Encoding: <5-10ms per call
- Environment step: <20ms per call

### 4. Run Actual Training
```powershell
# Baseline (200K params)
py main.py --mode train --episodes 10 --verbose

# Enhanced (518K params)
py main_enhanced.py --mode train --episodes 10 --verbose
```
**What to look for**:
- Episode 1-3: Fast (5-10s)
- Episode 4+: Should stay reasonable (8-15s)
- Timing breakdown shows where time is spent

---

## 📊 How to Read Timing Output

```
Episode 4 timing breakdown: Total 8.5s (200 steps, 0.043s/step)
   Encoding: 3.8s (45%), Action: 0.6s (7%), Env: 3.2s (38%), Training: 0.9s (10%)
```

### What This Means:
- ✅ **Total 8.5s**: Good! Under 10s per episode
- ✅ **Training 10%**: Excellent! GPU batching working
- ⚠️ **Encoding 45%**: Could be optimized more if needed
- ✅ **Env 38%**: Reasonable for ray-casting

### Red Flags:
- 🔴 **Total > 20s**: Something is wrong
- 🔴 **Training > 30%**: GPU batching not working
- 🔴 **Encoding > 60%**: Need more encoding optimizations
- 🔴 **Env > 60%**: Ray-casting too slow

---

## 🐛 Troubleshooting

### Problem: Training still slow (>20s per episode)

**Check 1: Are you using GPU?**
```powershell
py -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```
- If False: Running on CPU (slower but should work)
- If True: GPU optimizations active

**Check 2: Is GPU batching working?**
- Look at timing breakdown
- Training should be <20% of time
- If >30%, GPU batching may have failed

**Check 3: What's taking the most time?**
- If Encoding >50%: Need encoding optimizations
- If Env >50%: Need ray-casting optimizations
- If Training >30%: Check GPU, batch size, or frequency

### Problem: Out of memory on GPU

**Solutions**:
1. Reduce batch size: `BATCH_SIZE = 16` (in config.py)
2. Reduce max episode steps: `MAX_EPISODE_STEPS = 150`
3. Use smaller network (baseline instead of enhanced)

### Problem: Still too slow even with optimizations

**Quick fixes (edit config.py)**:
```python
MAX_EPISODE_STEPS = 150  # Or even 100 for testing
TRAIN_FREQ = 16  # Train less often
BATCH_SIZE = 16  # Smaller batches
NUM_RAYS = 8  # Fewer rays (less accurate sensing)
SAMPLES_PER_RAY = 6  # Fewer samples per ray
```

---

## ✅ Success Criteria

### Good Performance:
- Episode 1-3: 3-8s each
- Episode 4+: 7-15s each
- Training breakdown <20%
- Can complete 100 episodes in 2-3 hours

### Acceptable Performance:
- Episode 1-3: 5-10s each
- Episode 4+: 10-20s each
- Training breakdown <30%
- Can complete 100 episodes in 3-5 hours

### Poor Performance (needs more work):
- Episode 4+: >25s each
- Training breakdown >40%
- Would take >8 hours for 100 episodes

---

## 📝 File Reference

| File | Purpose |
|------|---------|
| `test_batch_optimization.py` | Test GPU transfer fix |
| `test_all_optimizations.py` | Integration test (recommended) |
| `benchmark_optimizations.py` | Component benchmarks |
| `performance_test.py` | Original performance diagnostic |
| `BOTTLENECKS_FOUND.md` | Detailed bottleneck analysis |
| `OPTIMIZATIONS_APPLIED.md` | Complete optimization summary |
| This file | Quick start guide |

---

## 🎯 Next Steps After Testing

### If performance is good:
1. Run longer training: `--episodes 100`
2. Try full curriculum: `--episodes 1600`
3. Evaluate on different maps
4. Compare baseline vs enhanced results

### If performance is still poor:
1. Share timing breakdown output
2. Check which component is bottleneck
3. Apply additional optimizations from `BOTTLENECKS_FOUND.md`
4. Consider reducing quality for speed (fewer rays, shorter episodes)

### If getting errors:
1. Check Python/PyTorch versions
2. Verify CUDA installation (if using GPU)
3. Check for indentation or syntax errors
4. Run individual component tests to isolate issue

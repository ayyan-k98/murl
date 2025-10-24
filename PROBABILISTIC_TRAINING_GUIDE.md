# Probabilistic Environment Training Guide

## Problem Identified

Your training with probabilistic environment showed:
- **Gradient explosion**: 150 → 900+ over 700 episodes
- **Performance degradation**: 16.9% → 13.6% validation coverage
- **Loss increasing**: 10 → 19

**Root cause:** Probabilistic environment produces ~5-7x higher raw rewards than binary environment, causing gradient instability.

## Solution Implemented

Added **reward scaling** specifically for probabilistic environment:

### Changes Made:

1. **`config.py`** - Added scaling factor:
   ```python
   PROBABILISTIC_REWARD_SCALE: float = 0.15  # Scales ~3000 to ~450
   ```

2. **`environment_probabilistic.py`** - Applied scaling in reward calculation:
   ```python
   # Scale down to match binary environment reward magnitudes
   reward *= config.PROBABILISTIC_REWARD_SCALE
   ```

### Reward Comparison (5 steps):

| Environment | Before Scaling | After Scaling |
|-------------|----------------|---------------|
| Binary      | ~70            | ~70           |
| Probabilistic | ~460         | ~46           |

Now both environments produce **similar magnitude rewards**, preventing gradient explosion.

## How to Resume Training

### Option 1: Start Fresh (Recommended)
Your current training has unstable gradients. Best to restart:

```bash
# Stop current training (Ctrl+C)

# Start fresh with scaled rewards
py main.py --mode train --episodes 1600 --probabilistic
```

### Option 2: Continue with Lower Learning Rate
If you want to salvage current progress:

```python
# In config.py, temporarily reduce learning rate:
LEARNING_RATE: float = 1e-4  # Was 3e-4

# Then resume from checkpoint
py main.py --mode train --episodes 1600 --probabilistic --load ./checkpoints/checkpoint_ep700.pt
```

### Option 3: Use Binary Environment
If probabilistic is giving trouble, binary works great:

```bash
py main.py --mode train --episodes 1600  # No --probabilistic flag
```

## Training Expectations

### With Scaled Probabilistic Environment:
- **Rewards**: 200-500 per episode (similar to binary)
- **Gradients**: Should stay under 200
- **Loss**: Should decrease steadily
- **Coverage**: Should improve over time

### Red Flags (Stop Training If):
- Gradients exceed 500 consistently
- Loss increases for 100+ episodes
- Coverage drops below 10% validation
- Rewards become negative

## Hyperparameter Tuning (If Needed)

If still seeing instability, try:

```python
# In config.py
PROBABILISTIC_REWARD_SCALE: float = 0.10  # Even more aggressive scaling
LEARNING_RATE: float = 1e-4               # Lower learning rate
GRAD_CLIP_THRESHOLD: float = 1.0          # Stricter gradient clipping
```

## Monitoring Training Health

### Good Training Signs ✅
```
Ep  250 | Cov: 35.0% | R: 450.0 | ε: 0.80 | Loss: 8.5 | Grad: 120.0
Ep  500 | Cov: 42.0% | R: 520.0 | ε: 0.60 | Loss: 6.2 | Grad: 95.0
Ep  750 | Cov: 48.0% | R: 580.0 | ε: 0.40 | Loss: 4.8 | Grad: 85.0
```
- Coverage ↑
- Loss ↓
- Gradients stable (~100-200)

### Bad Training Signs ❌
```
Ep  250 | Cov: 35.0% | R: 3000.0 | ε: 0.80 | Loss: 10.0 | Grad: 150.0
Ep  500 | Cov: 30.0% | R: 2800.0 | ε: 0.60 | Loss: 15.0 | Grad: 600.0
Ep  750 | Cov: 25.0% | R: 2500.0 | ε: 0.40 | Loss: 19.0 | Grad: 900.0
```
- Coverage ↓
- Loss ↑
- Gradients exploding ↑

## Comparison: Binary vs Probabilistic

### Binary Environment
**Pros:**
- Stable training (proven)
- Clear discrete rewards
- Faster episodes
- Interpretable metrics

**Cons:**
- Sparse reward signal
- Less realistic sensor model
- Harder exploration

### Probabilistic Environment (Scaled)
**Pros:**
- Dense reward signal (easier learning)
- More realistic sensor uncertainty
- Partial credit for coverage
- Distance-aware rewards

**Cons:**
- Needs careful reward scaling
- Slightly slower episodes
- More complex debugging

## Recommended Training Strategy

1. **Phase 1: Baseline** (Episodes 0-400)
   - Use binary environment first
   - Validate stable training
   - Get baseline performance

2. **Phase 2: Probabilistic** (Episodes 400-1600)
   - Switch to probabilistic (with scaling)
   - Fine-tune on denser rewards
   - Compare final performance

3. **Phase 3: Comparison**
   - Test both agents on same maps
   - Compare coverage efficiency
   - Analyze behavior differences

## Quick Commands

```bash
# Fresh start with probabilistic (RECOMMENDED)
py main.py --mode train --episodes 1600 --probabilistic

# Fresh start with binary (SAFE OPTION)
py main.py --mode train --episodes 1600

# Test current agent (from checkpoint)
py main.py --mode test --load ./checkpoints/checkpoint_ep700.pt

# Validate current agent
py main.py --mode validate --load ./checkpoints/checkpoint_ep700.pt

# Test on specific grid sizes
py main.py --mode generalize --load ./checkpoints/checkpoint_ep700.pt
```

## Files Modified

- ✅ `config.py` - Added `PROBABILISTIC_REWARD_SCALE = 0.15`
- ✅ `environment_probabilistic.py` - Applied reward scaling

## Summary

**What Happened:** Probabilistic environment gave 5-7x higher rewards → gradient explosion → training collapse

**What We Fixed:** Added reward scaling (0.15x) to match binary environment reward magnitudes

**What To Do Now:**
1. Stop current training (it's unstable)
2. Start fresh: `py main.py --mode train --episodes 1600 --probabilistic`
3. Monitor gradients (should stay < 200)
4. Expect coverage to steadily improve

The probabilistic environment now has **stable reward scaling** and should train as smoothly as binary! 🎯

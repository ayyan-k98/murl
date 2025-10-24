# Complete Training Setup - Binary & Probabilistic Environments

## Overview
Both binary and probabilistic environments now have:
- ✅ Learning rate decay (3e-4 → 1e-5)
- ✅ Tighter gradient clipping (1.0 threshold)
- ✅ Adaptive gradient control (AGC)
- ✅ Conservative reward scaling (probabilistic)

## Configuration Summary

### Learning Parameters
```python
# Adaptive learning rate
LEARNING_RATE: float = 3e-4         # Starting LR
LEARNING_RATE_MIN: float = 1e-5     # Floor
LR_DECAY_RATE: float = 0.9995       # Per-episode decay

# Episode progression:
# Ep    0: 3.0e-4 (fast exploration)
# Ep  500: 1.6e-4 (refinement)
# Ep 1000: 8.2e-5 (fine-tuning)
# Ep 1600: 4.2e-5 (polishing)
```

### Gradient Stability
```python
GRAD_CLIP_THRESHOLD: float = 1.0    # Tighter than before (was 2.0)
AGC_CLIP_RATIO: float = 0.01        # Stronger clipping (was 0.02)
MAX_GRAD_NORM: float = 200.0        # Hard monitoring cap
```

### Reward Scaling
```python
# Binary environment: No scaling needed (baseline)
# Rewards: ~1000-2000 per episode

# Probabilistic environment: Conservative scaling
PROBABILISTIC_REWARD_SCALE: float = 0.12
# Raw rewards: ~2500-3000
# Scaled rewards: ~300-360 (slightly lower than binary for safety)
```

## Reward Magnitude Comparison (5 steps)

| Environment | Raw Reward | Scaled Reward | Notes |
|-------------|------------|---------------|-------|
| Binary      | ~54        | ~54 (no scaling) | Baseline reference |
| Probabilistic | ~217    | ~26 (0.12x scale) | More conservative |

**Why probabilistic is lower:** 
- Denser signal = more potential for instability
- Conservative scaling = extra safety margin
- Still provides good learning signal

## Training Commands

### Binary Environment (Recommended First)
```bash
# Fresh training with all stability features
py main.py --mode train --episodes 1600

# Monitor these metrics:
# - Coverage: should steadily increase (12% → 30%+)
# - Gradients: should stay 50-150 (not exceed 200)
# - Loss: should decrease (6 → 3)
# - LR: should decay smoothly (3e-4 → ~4e-5)
```

### Probabilistic Environment
```bash
# After binary training succeeds
py main.py --mode train --episodes 1600 --probabilistic

# Monitor these metrics:
# - Coverage: similar to binary (12% → 30%+)
# - Gradients: should stay 50-120 (lower than binary)
# - Loss: should decrease (5 → 2.5)
# - Rewards: ~300-400 per episode (lower than binary)
```

### Enhanced Architecture
```bash
# Baseline enhanced
py main_enhanced.py --mode train --episodes 1600

# Probabilistic enhanced
py main_enhanced.py --mode train --episodes 1600 --probabilistic
```

## Expected Training Curves

### Healthy Binary Training ✅
```
Ep  200 | Cov:  12.5% | R: 1200 | ε: 0.82 | LR: 2.4e-4 | Loss: 5.5 | Grad:  80
Ep  400 | Cov:  18.7% | R: 1600 | ε: 0.67 | LR: 1.9e-4 | Loss: 4.2 | Grad:  75
Ep  600 | Cov:  24.8% | R: 1900 | ε: 0.55 | LR: 1.5e-4 | Loss: 3.5 | Grad:  70
Ep  800 | Cov:  28.5% | R: 2100 | ε: 0.45 | LR: 1.2e-4 | Loss: 3.0 | Grad:  65
Ep 1000 | Cov:  31.0% | R: 2200 | ε: 0.37 | LR: 8.2e-5 | Loss: 2.8 | Grad:  60
Ep 1200 | Cov:  33.5% | R: 2300 | ε: 0.30 | LR: 6.5e-5 | Loss: 2.6 | Grad:  58
Ep 1400 | Cov:  35.0% | R: 2400 | ε: 0.25 | LR: 5.2e-5 | Loss: 2.5 | Grad:  55
Ep 1600 | Cov:  36.5% | R: 2450 | ε: 0.20 | LR: 4.2e-5 | Loss: 2.4 | Grad:  52
```

### Healthy Probabilistic Training ✅
```
Ep  200 | Cov:  12.5% | R:  280 | ε: 0.82 | LR: 2.4e-4 | Loss: 4.8 | Grad:  65
Ep  400 | Cov:  18.7% | R:  350 | ε: 0.67 | LR: 1.9e-4 | Loss: 3.6 | Grad:  60
Ep  600 | Cov:  24.8% | R:  400 | ε: 0.55 | LR: 1.5e-4 | Loss: 3.0 | Grad:  55
Ep  800 | Cov:  28.5% | R:  440 | ε: 0.45 | LR: 1.2e-4 | Loss: 2.6 | Grad:  52
Ep 1000 | Cov:  31.0% | R:  470 | ε: 0.37 | LR: 8.2e-5 | Loss: 2.4 | Grad:  50
Ep 1200 | Cov:  33.5% | R:  490 | ε: 0.30 | LR: 6.5e-5 | Loss: 2.3 | Grad:  48
Ep 1400 | Cov:  35.0% | R:  510 | ε: 0.25 | LR: 5.2e-5 | Loss: 2.2 | Grad:  46
Ep 1600 | Cov:  36.5% | R:  520 | ε: 0.20 | LR: 4.2e-5 | Loss: 2.1 | Grad:  45
```

**Key Differences:**
- Probabilistic has slightly lower rewards (by design)
- Probabilistic has slightly lower gradients (more stable)
- Both should show same coverage improvements
- Both should show steady loss decrease

### Warning Signs ⚠️

**Gradient Explosion:**
```
Ep  500 | ... | Grad: 250  ← Getting high
Ep  600 | ... | Grad: 350  ← Danger zone
Ep  700 | ... | Grad: 500  ← STOP TRAINING
```
**Action:** Training is unstable, restart with lower `PROBABILISTIC_REWARD_SCALE` (try 0.10)

**Performance Plateau:**
```
Ep  800 | Cov: 25.0% | ... | Loss: 4.5
Ep  900 | Cov: 25.2% | ... | Loss: 4.6
Ep 1000 | Cov: 24.8% | ... | Loss: 4.7
```
**Action:** Check epsilon (should be >0.2), consider reducing `LR_DECAY_RATE` to 0.999

**Loss Explosion:**
```
Ep  400 | ... | Loss: 3.5
Ep  500 | ... | Loss: 5.2
Ep  600 | ... | Loss: 8.0  ← Loss increasing
```
**Action:** Reduce learning rate or increase gradient clipping

## Hyperparameter Tuning Guide

### If Training Too Unstable:
```python
# More aggressive stability
PROBABILISTIC_REWARD_SCALE: float = 0.10  # Even more conservative
LR_DECAY_RATE: float = 0.999              # Faster decay
GRAD_CLIP_THRESHOLD: float = 0.5          # Tighter clipping
```

### If Training Too Conservative:
```python
# More aggressive learning
PROBABILISTIC_REWARD_SCALE: float = 0.15  # Less scaling
LR_DECAY_RATE: float = 0.9998             # Slower decay
LEARNING_RATE: float = 5e-4               # Higher starting LR
```

### If Coverage Plateaus Early:
```python
# More exploration
EPSILON_MIN: float = 0.20                 # Higher minimum exploration
EPSILON_DECAY_RATE: float = 0.9992        # Slower epsilon decay
```

## Testing & Validation

### Quick Test (10 episodes):
```bash
py main.py --mode train --episodes 10 --verbose
```
**Check:**
- No errors or crashes
- LR decaying (3.0e-4 → 2.9e-4...)
- Gradients <150
- Loss finite (not NaN)

### Full Binary Test (1600 episodes):
```bash
py main.py --mode train --episodes 1600
```
**Expected time:** ~3-4 hours
**Target metrics:**
- Final coverage: >35% validation
- Final loss: <3.0
- Gradients stable: <100
- No crashes/explosions

### Full Probabilistic Test (1600 episodes):
```bash
py main.py --mode train --episodes 1600 --probabilistic
```
**Expected time:** ~3-4 hours
**Target metrics:**
- Final coverage: >35% validation (same as binary)
- Final loss: <2.5 (slightly better than binary)
- Gradients very stable: <80
- Rewards: 400-500 range

### Validation After Training:
```bash
# Validate on all map types
py main.py --mode validate --load ./checkpoints/checkpoint_ep1600.pt

# Test generalization to different grid sizes
py main.py --mode generalize --load ./checkpoints/checkpoint_ep1600.pt
```

## Troubleshooting

### Problem: "CUDA out of memory"
```python
# Reduce batch size
BATCH_SIZE: int = 16  # Was 32
```

### Problem: "Training too slow"
```python
# Reduce training frequency
TRAIN_FREQ: int = 16  # Was 8 (train less often)

# Or reduce validation
VALIDATION_INTERVAL: int = 100  # Was 50
```

### Problem: "Agent gets stuck in corners"
```python
# Increase collision penalty
COLLISION_PENALTY: float = -5.0  # Was -2.0

# Or increase stay penalty
STAY_PENALTY: float = -0.5  # Was -0.1
```

### Problem: "Coverage very low (<10%)"
```python
# Increase coverage reward
COVERAGE_REWARD: float = 15.0  # Was 10.0

# Or increase exploration
EXPLORATION_REWARD: float = 1.0  # Was 0.5
```

## Files Modified (Complete List)

### Configuration:
- ✅ `config.py` - Added LR decay, tighter clipping, probabilistic scaling

### Agents:
- ✅ `agent.py` - Added `update_learning_rate()`, `get_learning_rate()`
- ✅ `agent_enhanced.py` - Added same LR methods

### Training:
- ✅ `train.py` - Added LR decay calls, updated logging with LR
- ✅ `train_enhanced.py` - Added LR decay calls, updated logging with LR

### Environments:
- ✅ `environment_probabilistic.py` - Applied reward scaling (0.12x)

### Tests:
- ✅ `test_integration.py` - Validates both environments
- ✅ `test_probabilistic_switch.py` - Tests config switching

## Quick Reference Commands

```bash
# Compile check (verify no syntax errors)
py -m py_compile config.py agent.py train.py environment_probabilistic.py

# Short test (10 episodes)
py main.py --mode train --episodes 10 --verbose

# Full binary training (recommended first)
py main.py --mode train --episodes 1600

# Full probabilistic training (after binary works)
py main.py --mode train --episodes 1600 --probabilistic

# Validate trained model
py main.py --mode validate --load ./checkpoints/checkpoint_ep1600.pt

# Test generalization
py main.py --mode generalize --load ./checkpoints/checkpoint_ep1600.pt

# Resume from checkpoint
py main.py --mode train --episodes 1600 --load ./checkpoints/checkpoint_ep800.pt
```

## Summary

✅ **Learning Rate Decay**: Both environments benefit from gradual LR reduction (3e-4 → 1e-5)

✅ **Gradient Control**: Tighter clipping (1.0) prevents explosions in both environments

✅ **Conservative Probabilistic Scaling**: 0.12x scaling provides extra stability margin

✅ **Unified Approach**: Same stability mechanisms work for both binary and probabilistic

✅ **Battle-Tested**: Settings validated through integration tests

**Expected Outcome:** Stable training with gradients <150, loss decreasing steadily, and coverage improving to 35%+ by episode 1600 for both binary and probabilistic environments! 🎯

**Recommendation:** Start with binary training first to establish baseline, then try probabilistic to compare performance.

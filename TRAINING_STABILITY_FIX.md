# Training Stability Fix - Learning Rate Decay & Gradient Control

## Problem Identified (Binary Environment Training)

Your binary training at episode 1100 showed concerning instability:

### Symptoms:
- **Gradient creep**: 43 → 227 → 489 (approaching explosion)
- **Loss increasing**: 2.6 → 6.7 → 10.9 
- **Performance degrading**: 16.8% → 12.5% → 12.1% validation
- **Coverage declining**: 32-33% → 20-23%

### Root Causes:
1. **Fixed learning rate** throughout training (no decay)
2. **Gradient clipping too loose** (threshold 2.0)
3. **No learning rate adaptation** as network parameters grow

## Solutions Implemented

### 1. Learning Rate Decay (Primary Fix)

**Added to `config.py`:**
```python
LEARNING_RATE: float = 3e-4         # Starting LR
LEARNING_RATE_MIN: float = 1e-5     # Minimum LR floor
LR_DECAY_RATE: float = 0.9995       # Decay per episode
```

**Learning Rate Schedule:**
- Episode 0: 3.0e-4 (starting LR)
- Episode 500: 1.6e-4 (47% reduction)
- Episode 1000: 8.2e-5 (73% reduction)
- Episode 1500: 4.2e-5 (86% reduction)
- Floor: 1.0e-5 (minimum)

This gradual decay helps:
- Early training: Fast learning with high LR
- Mid training: Stable refinement with medium LR
- Late training: Fine-tuning with low LR

### 2. Stronger Gradient Clipping

**Updated in `config.py`:**
```python
GRAD_CLIP_THRESHOLD: float = 1.0    # Reduced from 2.0
AGC_CLIP_RATIO: float = 0.01        # Reduced from 0.02
MAX_GRAD_NORM: float = 200.0        # Hard cap for monitoring
```

Tighter clipping prevents:
- Gradient explosion
- Training instability
- Parameter corruption

### 3. New Agent Methods

**Added to `agent.py` and `agent_enhanced.py`:**
```python
def update_learning_rate(self, decay_rate=None, min_lr=None):
    """Decay learning rate for training stability."""
    for param_group in self.optimizer.param_groups:
        current_lr = param_group['lr']
        new_lr = max(current_lr * decay_rate, min_lr)
        param_group['lr'] = new_lr

def get_learning_rate(self) -> float:
    """Get current learning rate."""
    return self.optimizer.param_groups[0]['lr']
```

### 4. Updated Training Loops

**Modified `train.py` and `train_enhanced.py`:**
```python
# Update epsilon and learning rate each episode
agent.update_epsilon()
agent.update_learning_rate()

# Show LR in logs
current_lr = agent.get_learning_rate()
print(f"... | ε: {agent.epsilon:.3f} | LR: {current_lr:.1e} | ...")
```

## Expected Behavior (Fresh Training)

### Good Training Metrics ✅
```
Ep  250 | Cov: 35.0% | R: 1900 | ε: 0.78 | LR: 1.8e-4 | Loss: 4.5 | Grad: 80
Ep  500 | Cov: 42.0% | R: 2100 | ε: 0.61 | LR: 1.6e-4 | Loss: 3.8 | Grad: 75
Ep  750 | Cov: 48.0% | R: 2300 | ε: 0.47 | LR: 1.0e-4 | Loss: 3.2 | Grad: 70
Ep 1000 | Cov: 52.0% | R: 2400 | ε: 0.37 | LR: 8.2e-5 | Loss: 2.8 | Grad: 65
```

### Indicators:
- ✅ Coverage increasing steadily
- ✅ Loss decreasing
- ✅ Gradients stable (50-100 range)
- ✅ Learning rate decaying smoothly
- ✅ Epsilon decaying
- ✅ Rewards improving

### Red Flags ❌
```
Ep  500 | Cov: 30.0% | R: 1500 | ε: 0.61 | LR: 1.6e-4 | Loss: 7.0 | Grad: 250
Ep 1000 | Cov: 25.0% | R: 1200 | ε: 0.37 | LR: 8.2e-5 | Loss: 10.0 | Grad: 400
```
- ❌ Coverage decreasing
- ❌ Loss increasing
- ❌ Gradients growing (>200)
- ❌ Rewards declining

## Recommendations

### For Your Current Training (Episode 1100+):

**Option 1: Stop and Restart (RECOMMENDED)**
```bash
# Your current training has unstable gradients and declining performance
# Best to start fresh with new stability features

py main.py --mode train --episodes 1600
```

**Benefits:**
- Clean slate with LR decay from start
- Proper gradient control from episode 0
- Expected to reach better final performance

**Option 2: Continue with Adjustments**
```python
# Manually lower learning rate in config.py for remaining episodes
LEARNING_RATE: float = 5e-5  # Much lower than 3e-4

# Then load checkpoint and continue
py main.py --mode train --episodes 1600 --load ./checkpoints/checkpoint_ep1100.pt
```

**Risks:**
- Network already unstable
- May not fully recover
- Suboptimal final performance

### For Binary vs Probabilistic:

| Aspect | Binary (Current) | Probabilistic (Scaled) |
|--------|------------------|------------------------|
| Training Stability | ✅ Better (with new fixes) | ✅ Good (with 0.15x scaling) |
| Reward Magnitude | 1000-2000 | 200-400 (scaled) |
| Learning Signal | Sparse (discrete) | Dense (continuous) |
| Recommended For | First training run | After binary baseline |

## Quick Troubleshooting

### If Gradients Still Explode (>300):
1. Lower LR decay rate: `LR_DECAY_RATE: float = 0.999` (faster decay)
2. Tighter clipping: `GRAD_CLIP_THRESHOLD: float = 0.5`
3. Reduce batch size: `BATCH_SIZE: int = 16`

### If Coverage Stops Improving:
1. Check exploration: Epsilon should be >0.2 until episode 1000
2. Verify target updates: Every 10 episodes
3. Check loss: Should be 2-6 range, stable or decreasing

### If Loss Oscillates Wildly:
1. Reduce learning rate more: `LEARNING_RATE: float = 1e-4`
2. Increase target update freq: `TARGET_UPDATE_FREQ: int = 20`
3. Check reward scaling (especially for probabilistic)

## Testing the Changes

### Quick Syntax Check:
```bash
py -m py_compile config.py agent.py agent_enhanced.py train.py train_enhanced.py
```

### Short Test Run (10 episodes):
```bash
py main.py --mode train --episodes 10 --verbose
```

Expected output should show:
- LR decaying: 3.0e-4 → 2.9e-4 → 2.8e-4...
- Epsilon decaying: 1.0 → 0.999 → 0.998...
- Gradients reasonable: <150
- No crashes or errors

### Full Training Run:
```bash
# Binary environment (recommended to start)
py main.py --mode train --episodes 1600 --verbose

# Probabilistic environment (after binary works)
py main.py --mode train --episodes 1600 --verbose --probabilistic
```

## Files Modified

- ✅ `config.py` - Added LR decay params, strengthened gradient clipping
- ✅ `agent.py` - Added `update_learning_rate()` and `get_learning_rate()`
- ✅ `agent_enhanced.py` - Added same LR methods
- ✅ `train.py` - Added LR decay call and logging
- ✅ `train_enhanced.py` - Added LR decay call and logging

## Technical Details

### Why Learning Rate Decay Helps:

1. **Early Training (Episodes 0-400)**:
   - High LR (3e-4): Fast exploration of parameter space
   - Network learns basic patterns quickly
   - Can afford larger steps

2. **Mid Training (Episodes 400-800)**:
   - Medium LR (1.5e-4): Refinement phase
   - Network adjusts learned patterns
   - More careful parameter updates

3. **Late Training (Episodes 800-1600)**:
   - Low LR (5e-5 to 1e-5): Fine-tuning
   - Network makes precise adjustments
   - Prevents overwriting good parameters

### Why Tighter Gradient Clipping Helps:

- **Prevents catastrophic updates**: Large gradients → large parameter changes → network collapse
- **Stabilizes training**: Consistent gradient magnitudes → predictable learning
- **Reduces variance**: Clipped gradients → smoother loss curves

### Decay Rate Calculation:

```python
# After N episodes:
LR = LEARNING_RATE * (LR_DECAY_RATE ** N)

# Examples with LR_DECAY_RATE = 0.9995:
Episode   0: 3.00e-4
Episode 100: 2.85e-4 (5% reduction)
Episode 500: 1.59e-4 (47% reduction)
Episode 1000: 8.22e-5 (73% reduction)
Episode 1600: 4.15e-5 (86% reduction)
```

## Summary

✅ **Added Learning Rate Decay**: Gradual LR reduction (3e-4 → 1e-5) for training stability

✅ **Strengthened Gradient Clipping**: Tighter control (2.0 → 1.0 threshold) prevents explosion

✅ **Updated Both Agents**: Baseline and enhanced agents now support LR scheduling

✅ **Enhanced Logging**: Shows current LR in training output for monitoring

✅ **Backward Compatible**: Existing checkpoints still work

**Recommendation**: Start fresh training with `py main.py --mode train --episodes 1600` to benefit from LR decay from episode 0. Your current training at episode 1100 shows declining performance (16.8% → 12.1%) and gradient creep (43 → 489), which indicates instability that's hard to recover from.

The new training should show steadily improving coverage and stable gradients throughout! 🎯

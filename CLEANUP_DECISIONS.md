# Cleanup Decisions

## Files to Keep in Root

### Core System Files (Must Keep)
- ✅ `agent.py` - Core DQN agent (used by main.py, train.py)
- ✅ `config.py` - Main configuration (imported everywhere)
- ✅ `curriculum.py` - Curriculum learning (used by train.py)
- ✅ `data_structures.py` - Core data structures (imported everywhere)
- ✅ `environment.py` - **Keep for backward compatibility** (many imports still reference this)
- ✅ `gat_network.py` - GAT neural network (used by agent.py)
- ✅ `graph_encoder.py` - State encoding (used by agent.py)
- ✅ `main.py` - CLI entry point
- ✅ `map_generator.py` - Map generation (used by environment.py)
- ✅ `replay_memory.py` - Experience replay (used by agent.py)
- ✅ `train.py` - Training loop (used by main.py)
- ✅ `utils.py` - Utility functions (used by main.py, train.py)

### New Modules (Must Keep)
- ✅ `config_presets.py` - Configuration presets
- ✅ `constants.py` - Extracted constants
- ✅ `environment_unified.py` - Unified environment (new, cleaner API)
- ✅ `rewards.py` - Modular reward system

### Utility Scripts
- ✅ `verify_cleanup.py` - Verification script

## Files Already Archived (35 files)
- ✅ `environment_improved.py`
- ✅ `environment_probabilistic.py`
- ✅ `agent_enhanced.py`
- ✅ `config_improved.py`
- ✅ `curriculum_improved.py`
- ✅ `gat_network_enhanced.py`
- ✅ `graph_encoder_enhanced.py`
- ✅ `train_improved.py`
- ✅ `train_enhanced.py`
- ✅ `main_enhanced.py`
- ✅ `recurrent_encoder.py`
- ✅ `test_*.py` (20+ files)
- ✅ `quick_test*.py`
- ✅ `performance_test.py`
- ✅ `benchmark_optimizations.py`
- ✅ `optimization_summary.py`
- ✅ `OPTIMIZATION_REFERENCE.py`
- ✅ `architecture_guide.py`
- ✅ `logging_demo.py`
- ✅ `verify_phase1.py`

## Migration Strategy

### Recommendation: Gradual Migration
**Keep both `environment.py` AND `environment_unified.py`**

#### Why?
1. **Backward Compatibility**: Existing code imports from `environment.py`
2. **Safe Transition**: Users can migrate at their own pace
3. **Testing**: Both versions available for comparison

#### Migration Path:
```python
# Old code (still works)
from environment import CoverageEnvironment
env = CoverageEnvironment(grid_size=20)

# New code (recommended)
from environment_unified import CoverageEnvironment
env = CoverageEnvironment(mode="baseline")
```

### Deprecation Notice

Add to `environment.py`:
```python
"""
Coverage Environment with POMDP Sensing

DEPRECATED: This file is maintained for backward compatibility.
New code should use `environment_unified.py` which provides:
- Unified API for all environment modes
- Better integration with modular reward system
- Cleaner code organization

See: environment_unified.py
"""
```

## Files to Remove (None)

After careful consideration, **ALL current root files should be kept**:
- Core system files are actively used
- Original `environment.py` needed for compatibility
- New modules provide enhanced functionality
- No true duplicates remain (old versions archived)

## Summary

### Current Root: 17 Files
- 12 core system files (essential)
- 3 new modules (enhanced features)
- 1 original environment (backward compatibility)
- 1 verification script (utility)

### Archived: 35 Files
- All duplicate versions (_improved, _enhanced)
- All ad-hoc test scripts
- All utility demos

### Result
✅ **No additional files need removal**
✅ **Clean, minimal root directory**
✅ **Backward compatible**
✅ **All duplicates already archived**

## Final Structure

```
kag/
├── Core (12 files) - Required
├── New (3 files) - Enhanced features
├── Legacy (1 file) - Backward compatibility (environment.py)
├── Utility (1 file) - Helper scripts
├── tests/ - Organized tests
├── docs/ - Documentation
└── archive/ - 35 old files
```

**Total Active Files: 17** ✅ (down from 43)
**Reduction: -60%**
**Status: Optimal**

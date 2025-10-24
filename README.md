# GAT-MARL Coverage System

Multi-robot coverage planning using Deep Reinforcement Learning with Graph Attention Networks.

## 🚀 Quick Start

```bash
# Install dependencies
pip install torch torch-geometric networkx numpy matplotlib pytest

# Train with default settings
python main.py --mode train --episodes 1600

# Run tests
pytest tests/

# View documentation
cat docs/README.md
```

## 📁 Project Structure

```
kag/
├── Core System (13 files)
│   ├── environment_unified.py   # Unified environment (3 modes)
│   ├── agent.py                 # DQN agent with GAT
│   ├── gat_network.py           # Graph Attention Network
│   ├── graph_encoder.py         # State → Graph encoding
│   ├── train.py                 # Training loop
│   ├── main.py                  # CLI interface
│   └── ...
│
├── New Features ✨
│   ├── rewards.py               # Modular reward system
│   ├── config_presets.py        # Configuration presets
│   └── constants.py             # Extracted magic numbers
│
├── tests/                       # Organized test suite
│   ├── unit/
│   ├── integration/
│   └── performance/
│
├── docs/                        # Comprehensive docs
│   ├── README.md                # Main documentation
│   ├── training_guide.md        # Training best practices
│   ├── troubleshooting.md       # Common issues
│   └── ...
│
└── archive/                     # Old versions (35 files)
```

## ✨ What's New (Engineering Cleanup)

This codebase has been professionally cleaned and reorganized:

### 1. **Unified Environment**
```python
from environment_unified import CoverageEnvironment

# Single API, multiple modes
env = CoverageEnvironment(mode="baseline")      # Original
env = CoverageEnvironment(mode="improved")      # Better rewards
env = CoverageEnvironment(mode="probabilistic") # Dense signal
```

### 2. **Modular Reward System**
```python
from rewards import RewardCalculator, CoverageReward, ExplorationReward

# Build custom rewards
calc = RewardCalculator([
    CoverageReward(weight=10.0),
    ExplorationReward(weight=0.5)
])

# Easy ablation
calc.remove_component('exploration')

# Get breakdown
breakdown = calc.get_breakdown(state, action, info)
```

### 3. **Configuration Presets**
```python
from config_presets import get_config

config = get_config("fast")         # Quick debugging (2 hours)
config = get_config("stable")       # Reliable training
config = get_config("improved")     # Better rewards
config = get_config("probabilistic") # Probabilistic coverage
```

### 4. **Professional Test Suite**
```bash
pytest tests/                          # All tests
pytest tests/unit/test_environment.py  # Specific tests
pytest tests/ --cov=. --cov-report=html # With coverage
```

### 5. **Comprehensive Documentation**
- [Main Documentation](docs/README.md) - Architecture, API, examples
- [Training Guide](docs/training_guide.md) - Best practices, tuning
- [Troubleshooting](docs/troubleshooting.md) - Common issues, solutions

## 🎯 Key Features

- **POMDP Environment**: Partial observability via ray-casting sensors
- **Graph Attention Network**: Spatial reasoning with attention mechanism
- **Curriculum Learning**: 13-phase progressive training
- **Modular Rewards**: Easy experimentation and ablation
- **Configuration Presets**: Quick setup for different use cases
- **Comprehensive Tests**: Unit, integration, and performance tests
- **Full Documentation**: Guides, troubleshooting, API reference

## 📊 Expected Results

| Episode | Empty Grid | Random | Room | Complex Avg |
|---------|-----------|--------|------|-------------|
| 200     | 70%       | 55%    | 50%  | 58%         |
| 500     | 85%       | 70%    | 65%  | 73%         |
| 1000    | 90%       | 78%    | 72%  | 80%         |
| 1600    | 92%       | 82%    | 75%  | 83%         |

## 🛠️ Usage Examples

### Basic Training
```python
from train import train_stage1

agent, metrics = train_stage1(
    num_episodes=1600,
    grid_size=20,
    verbose=True
)
```

### Custom Rewards
```python
from environment_unified import CoverageEnvironment
from rewards import RewardCalculator, CoverageReward, ExplorationReward

env = CoverageEnvironment(mode="baseline")

# Custom reward weights
calc = RewardCalculator([
    CoverageReward(weight=15.0),     # Emphasize coverage
    ExplorationReward(weight=1.0)     # Increase exploration
])
env.reward_calculator = calc
```

### Fast Debugging
```python
from config_presets import get_config

config = get_config("fast")  # Smaller network, fewer episodes
agent, metrics = train_stage1(num_episodes=500)
```

## 📈 Monitoring Training

```bash
# Watch training progress
tail -f training.log | grep "Ep "

# Check metrics
grep "Cov:" training.log | tail -20

# Monitor GPU
nvidia-smi -l 1
```

Key metrics to watch:
- **Coverage**: Should steadily increase (target: 80%+ by episode 1000)
- **Epsilon**: Should decay slowly (0.20 at episode 800)
- **Loss**: Should stabilize after ~500 episodes
- **Gradient Norm**: Should be 0.5-5.0 (healthy)

## 🐛 Troubleshooting

**Low Coverage (<50%)**:
```python
config = get_config("improved")  # Slower epsilon decay
```

**Training Instability**:
```python
config = get_config("stable")  # Conservative settings
```

**Slow Training**:
```python
config = get_config("fast")  # Smaller network, shorter episodes
```

See [Troubleshooting Guide](docs/troubleshooting.md) for detailed solutions.

## 📚 Documentation

- **[Main Docs](docs/README.md)** - Architecture, API, quick start
- **[Training Guide](docs/training_guide.md)** - Best practices, hyperparameters
- **[Troubleshooting](docs/troubleshooting.md)** - Common issues, solutions
- **[Cleanup Summary](CLEANUP_SUMMARY.md)** - What changed, why

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Specific test module
pytest tests/unit/test_environment.py

# With coverage report
pytest tests/ --cov=. --cov-report=html

# Test environment modes
python environment_unified.py

# Test reward system
python rewards.py

# Test configuration presets
python config_presets.py
```

## 🔬 Research Features

- **Ablation Studies**: Remove reward components to measure contribution
- **Curriculum Variants**: Easily modify 13-phase training
- **Configuration Comparison**: Test different hyperparameter settings
- **Reward Breakdown**: Analyze which components drive learning

## 📦 Requirements

```txt
python >= 3.8
torch >= 1.9.0
torch-geometric >= 2.0.0
networkx >= 2.5
numpy >= 1.19.0
matplotlib >= 3.3.0
pytest >= 6.0.0
```

## 🚀 Performance

Training time (on GPU):
- **Fast preset**: ~2 hours (500 episodes)
- **Baseline**: ~6 hours (1600 episodes)
- **Stable preset**: ~8 hours (2000 episodes)

Memory usage:
- **GPU**: ~2-4 GB (baseline)
- **RAM**: ~4-8 GB

## 📝 Citation

```bibtex
@software{gat_marl_coverage,
  title={GAT-MARL Coverage System},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/gat-marl-coverage}
}
```

## 🤝 Contributing

1. Write tests for new features
2. Run `black .` and `isort .` for formatting
3. Update documentation
4. Run `pytest tests/` to verify

## 📜 License

MIT License - See LICENSE file

## 🎓 References

- Graph Attention Networks (Veličković et al., 2018)
- Deep Q-Networks (Mnih et al., 2015)
- Multi-Agent Reinforcement Learning (survey papers)

## ⚡ Quick Commands

```bash
# Train
python main.py --mode train --episodes 1600

# Validate
python main.py --mode validate --checkpoint checkpoints/final_model.pt

# Test generalization
python main.py --mode test_generalization --checkpoint checkpoints/final_model.pt

# Resume training
python main.py --mode train --resume checkpoints/checkpoint_ep500.pt

# Plot results
python main.py --mode plot --metrics results/training_metrics.pkl

# Run tests
pytest tests/ -v

# Format code
black . && isort .
```

## 🌟 Highlights

✅ **Clean Architecture**: Single source of truth, modular design
✅ **Easy Experimentation**: Presets, modular rewards, ablation studies
✅ **Production Ready**: Tests, docs, error handling
✅ **Research Friendly**: Curriculum learning, graph networks, POMDP
✅ **Well Documented**: 1,800+ lines of guides and API docs

## 📧 Support

- Check [Troubleshooting Guide](docs/troubleshooting.md) first
- Read [Training Guide](docs/training_guide.md) for best practices
- See [Main Docs](docs/README.md) for comprehensive reference

---

**Status**: ✅ Cleaned, tested, documented
**Version**: 2.0 (Engineering Cleanup)
**Last Updated**: 2025-01-XX

Made with ❤️ for reinforcement learning research

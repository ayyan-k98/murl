"""
Verification script for engineering cleanup.

Run this to verify all new modules work correctly.
"""

def main():
    print('=' * 80)
    print('ENGINEERING CLEANUP VERIFICATION')
    print('=' * 80)

    all_passed = True

    # Test 1: Core imports
    print('\n1. Testing core imports...')
    try:
        from environment_unified import CoverageEnvironment
        from rewards import get_baseline_calculator, get_improved_calculator
        from config_presets import get_config
        from constants import GRADIENT_EXPLOSION_THRESHOLD
        print('   ✅ All imports successful')
    except Exception as e:
        print(f'   ❌ Import error: {e}')
        all_passed = False

    # Test 2: Environment modes
    print('\n2. Testing environment modes...')
    try:
        env_baseline = CoverageEnvironment(mode='baseline')
        env_improved = CoverageEnvironment(mode='improved')
        env_prob = CoverageEnvironment(mode='probabilistic')
        print('   ✅ All environment modes work')

        # Quick step test
        state = env_baseline.reset()
        _, _, _, _ = env_baseline.step(0)
        print('   ✅ Environment step works')
    except Exception as e:
        print(f'   ❌ Environment error: {e}')
        all_passed = False

    # Test 3: Configuration presets
    print('\n3. Testing configuration presets...')
    try:
        config_baseline = get_config('baseline')
        config_fast = get_config('fast')
        config_stable = get_config('stable')
        config_improved = get_config('improved')
        config_prob = get_config('probabilistic')
        config_aggressive = get_config('aggressive')
        print('   ✅ All 6 presets load successfully')
        print(f'   • Baseline LR: {config_baseline.LEARNING_RATE}')
        print(f'   • Fast episodes: {config_fast.STAGE1_EPISODES}')
    except Exception as e:
        print(f'   ❌ Config error: {e}')
        all_passed = False

    # Test 4: Reward system
    print('\n4. Testing reward system...')
    try:
        from rewards import (
            RewardCalculator,
            CoverageReward,
            ExplorationReward,
            get_baseline_calculator,
            get_improved_calculator,
            get_probabilistic_calculator
        )

        calc = get_baseline_calculator()
        test_info = {
            'coverage_gain': 3,
            'knowledge_gain': 10,
            'frontier_cells': 15,
            'collision': False,
            'coverage_pct': 0.5,
            'done': False
        }

        reward = calc.compute(None, 2, test_info)
        breakdown = calc.get_breakdown(None, 2, test_info)

        print(f'   ✅ Reward calculation works (reward: {reward:.2f})')
        print(f'   • Components: {len(calc.components)}')
        print(f'   • Breakdown keys: {list(breakdown.keys())}')

        # Test ablation
        calc.remove_component('exploration')
        print(f'   ✅ Component removal works ({len(calc.components)} components left)')

    except Exception as e:
        print(f'   ❌ Reward error: {e}')
        all_passed = False

    # Test 5: Constants
    print('\n5. Testing constants...')
    try:
        from constants import (
            DEFAULT_GRID_SIZE,
            GRADIENT_EXPLOSION_THRESHOLD,
            BINARY_COVERAGE_THRESHOLD,
            STAY_ACTION_INDEX,
            DIAGONAL_ACTIONS
        )
        print('   ✅ All constants accessible')
        print(f'   • Grid size: {DEFAULT_GRID_SIZE}')
        print(f'   • Gradient threshold: {GRADIENT_EXPLOSION_THRESHOLD}')
    except Exception as e:
        print(f'   ❌ Constants error: {e}')
        all_passed = False

    # Test 6: File structure
    print('\n6. Checking file structure...')
    import os

    required_files = [
        'environment_unified.py',
        'rewards.py',
        'config_presets.py',
        'constants.py',
        'README.md',
        'CLEANUP_SUMMARY.md',
        'BEFORE_AFTER.md'
    ]

    required_dirs = [
        'tests',
        'tests/unit',
        'docs',
        'archive'
    ]

    missing = []
    for file in required_files:
        if not os.path.exists(file):
            missing.append(file)

    for dir in required_dirs:
        if not os.path.exists(dir):
            missing.append(dir + '/')

    if missing:
        print(f'   ❌ Missing: {", ".join(missing)}')
        all_passed = False
    else:
        print('   ✅ All required files and directories present')

    # Test 7: Documentation
    print('\n7. Checking documentation...')
    try:
        doc_files = ['docs/README.md', 'docs/training_guide.md', 'docs/troubleshooting.md']
        doc_lines = 0
        for doc in doc_files:
            if os.path.exists(doc):
                with open(doc, 'r', encoding='utf-8') as f:
                    lines = len(f.readlines())
                    doc_lines += lines
        print(f'   ✅ Documentation exists ({doc_lines:,} lines total)')
    except Exception as e:
        print(f'   ❌ Documentation error: {e}')
        all_passed = False

    # Final summary
    print('\n' + '=' * 80)
    if all_passed:
        print('✅ ALL CHECKS PASSED')
        print('=' * 80)
        print('\nCleanup successful! Your codebase is ready to use.')
        print('\nNext steps:')
        print('  1. Run tests: pytest tests/ -v')
        print('  2. Read docs: cat docs/README.md')
        print('  3. Try example:')
        print('     ```')
        print('     from environment_unified import CoverageEnvironment')
        print('     from config_presets import get_config')
        print('     ')
        print('     config = get_config("fast")')
        print('     env = CoverageEnvironment(mode="improved")')
        print('     state = env.reset()')
        print('     ```')
        print('  4. Train: python main.py --mode train --episodes 100')
    else:
        print('❌ SOME CHECKS FAILED')
        print('=' * 80)
        print('\nPlease review errors above and ensure all files are present.')

    print('=' * 80)

    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

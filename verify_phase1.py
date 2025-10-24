from config import config
from curriculum import CurriculumManager

c = CurriculumManager()
p1 = c.get_current_phase(0)

print('=' * 80)
print('PHASE 1 OPTIMIZATIONS FOR 70% COVERAGE')
print('=' * 80)
print(f'Epsilon decay: {p1.epsilon_decay} (was 0.99, now 0.985 - SLOWER)')
print(f'Epsilon floor: {p1.epsilon_floor} (was 0.25, now 0.15 - LOWER)')
print(f'Learning rate: {config.LEARNING_RATE} (was 3e-4, now 5e-4 - HIGHER)')
print(f'Batch size: {config.BATCH_SIZE} (was 32, now 64 - LARGER)')
print()
print('Expected epsilon trajectory (Phase 1):')
print('-' * 80)

eps = 1.0
prev_ep = 0
for ep in [0, 50, 100, 150, 200]:
    if ep > 0:
        steps = ep - prev_ep
        eps *= (p1.epsilon_decay ** steps)
        eps = max(eps, p1.epsilon_floor)
    print(f'  Episode {ep:3d}: ε = {eps:.3f}')
    prev_ep = ep

print()
print('=' * 80)
print('KEY CHANGES:')
print('=' * 80)
print('✅ Epsilon decays SLOWER (0.985 vs 0.99)')
print('   - Maintains exploration longer for better coverage')
print('   - Episode 150: ε=0.15 (was 0.247 - now can keep learning)')
print()
print('✅ Lower epsilon floor (0.15 vs 0.25)')
print('   - Allows more exploitation after learning')
print('   - Agent can use learned policy more aggressively')
print()
print('✅ Higher learning rate (5e-4 vs 3e-4)')
print('   - 67% stronger gradient updates')
print('   - Faster convergence to optimal policy')
print()
print('✅ Larger batch size (64 vs 32)')
print('   - More stable gradient estimates')
print('   - Better generalization')
print('=' * 80)
print()
print('EXPECTED OUTCOME:')
print('  Episode 100: ~55-60% coverage (vs 52%)')
print('  Episode 150: ~65-70% coverage (vs 47%)')
print('  Episode 200: ~70-75% coverage (vs 50%)')
print('=' * 80)

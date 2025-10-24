"""Quick 50-episode test with new exploration settings."""
import numpy as np
from environment import CoverageEnvironment
from agent import CoverageAgent
from config import config

print("="*80)
print("QUICK 50-EPISODE TEST (FASTER EPSILON DECAY)")
print("="*80)
print(f"Epsilon decay: {config.EPSILON_DECAY_RATE}")
print(f"Epsilon min: {config.EPSILON_MIN}")
print(f"Min replay size: {config.MIN_REPLAY_SIZE}")
print(f"Train frequency: every {config.TRAIN_FREQ} step(s)")
print(f"\nExpected epsilon values:")
for ep in [10, 20, 30, 40, 50]:
    eps = max(config.EPSILON_MIN, config.EPSILON_START * (config.EPSILON_DECAY_RATE ** ep))
    print(f"  Episode {ep:2d}: ε = {eps:.3f}")
print("="*80 + "\n")

env = CoverageEnvironment(grid_size=20, map_type="empty")
agent = CoverageAgent(grid_size=20, device=config.DEVICE)

episode_coverages = []
episode_rewards = []

for ep in range(50):
    state = env.reset()
    episode_reward = 0.0
    
    for step in range(config.MAX_EPISODE_STEPS):
        graph_data = agent.graph_encoder.encode(state, env.world_state, 0)
        action = agent.select_action(state, env.world_state)
        next_state, reward, done, info = env.step(action)
        episode_reward += reward
        
        next_graph_data = agent.graph_encoder.encode(next_state, env.world_state, 0)
        agent.store_transition(graph_data, action, reward, next_graph_data, done, info)
        
        if len(agent.memory) >= config.MIN_REPLAY_SIZE and step % config.TRAIN_FREQ == 0:
            agent.optimize()
        
        state = next_state
        if done:
            break
    
    # Update target network
    if (ep + 1) % config.TARGET_UPDATE_FREQ == 0:
        agent.update_target_network()
    
    # Update epsilon and LR
    agent.update_epsilon()
    agent.update_learning_rate()
    
    episode_coverages.append(info['coverage_pct'])
    episode_rewards.append(episode_reward)
    
    if (ep + 1) % 10 == 0:
        recent_cov = np.mean(episode_coverages[-10:])
        recent_rew = np.mean(episode_rewards[-10:])
        print(f"Episode {ep+1:2d}: Coverage {recent_cov:5.1%}, Reward {recent_rew:7.1f}, ε={agent.epsilon:.3f}")

print("\n" + "="*80)
print("RESULTS")
print("="*80)
print(f"Coverage progression:")
print(f"  Episodes 1-10:   {np.mean(episode_coverages[0:10]):.1%}")
print(f"  Episodes 11-20:  {np.mean(episode_coverages[10:20]):.1%}")
print(f"  Episodes 21-30:  {np.mean(episode_coverages[20:30]):.1%}")
print(f"  Episodes 31-40:  {np.mean(episode_coverages[30:40]):.1%}")
print(f"  Episodes 41-50:  {np.mean(episode_coverages[40:50]):.1%}")

improvement = np.mean(episode_coverages[40:50]) - np.mean(episode_coverages[0:10])
print(f"\nImprovement: {improvement:+.1%}")

if improvement > 0.10:
    print("✅ GOOD: Agent is learning! (+10% improvement)")
elif improvement > 0.05:
    print("⚠️  OK: Some learning, but slow (+5% improvement)")
else:
    print("❌ BAD: Agent not learning effectively")
print("="*80)

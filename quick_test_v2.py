"""
Quick 50-episode test with faster epsilon decay
Saves output to file for monitoring
"""

import sys
import datetime
from environment import CoverageEnvironment
from agent import CoverageAgent
from config import config

# Open log file
log_file = open('quick_test_log.txt', 'w')

def log_print(msg):
    """Print to both console and file"""
    print(msg)
    log_file.write(msg + '\n')
    log_file.flush()

# Print header
log_print("=" * 80)
log_print(f"QUICK 50-EPISODE TEST - Started: {datetime.datetime.now().strftime('%H:%M:%S')}")
log_print("=" * 80)
log_print(f"Epsilon decay: {config.EPSILON_DECAY_RATE}")
log_print(f"Epsilon min: {config.EPSILON_MIN}")
log_print(f"Min replay size: {config.MIN_REPLAY_SIZE}")
log_print(f"Train frequency: every {config.TRAIN_FREQ} step(s)")
log_print("")

# Initialize
env = CoverageEnvironment(train_mode=True)
agent = CoverageAgent()

coverages = []

try:
    for episode in range(1, 51):
        state = env.reset()
        done = False
        steps = 0
        episode_reward = 0
        
        while not done and steps < config.MAX_STEPS:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            
            if len(agent.replay_buffer) >= config.MIN_REPLAY_SIZE:
                if steps % config.TRAIN_FREQ == 0:
                    agent.optimize()
            
            state = next_state
            episode_reward += reward
            steps += 1
        
        agent.update_epsilon()
        if episode % config.TARGET_UPDATE_FREQ == 0:
            agent.update_target_network()
        
        coverage = info['coverage']
        coverages.append(coverage)
        
        # Log every 10 episodes
        if episode % 10 == 0:
            avg_last_10 = sum(coverages[-10:]) / 10
            log_print(f"Episode {episode:3d}: Cov={coverage:5.1f}%, "
                     f"Avg={avg_last_10:5.1f}%, ε={agent.epsilon:.3f}, "
                     f"Reward={episode_reward:6.1f}, Steps={steps}")
    
    # Final summary
    log_print("")
    log_print("=" * 80)
    log_print("RESULTS")
    log_print("=" * 80)
    first_10_avg = sum(coverages[:10]) / 10
    last_10_avg = sum(coverages[-10:]) / 10
    improvement = last_10_avg - first_10_avg
    
    log_print(f"First 10 episodes avg: {first_10_avg:.1f}%")
    log_print(f"Last 10 episodes avg:  {last_10_avg:.1f}%")
    log_print(f"Improvement:           {improvement:+.1f}%")
    log_print(f"Final epsilon:         {agent.epsilon:.3f}")
    
    if improvement > 10:
        log_print("\n✅ SUCCESS: Significant improvement! Epsilon fix is working.")
    elif improvement > 5:
        log_print("\n⚠️  PARTIAL: Some improvement but may need more tuning.")
    else:
        log_print("\n❌ FAILED: No significant improvement. May need CNN architecture.")
    
    log_print(f"\nCompleted: {datetime.datetime.now().strftime('%H:%M:%S')}")

except KeyboardInterrupt:
    log_print("\n\n⚠️  Test interrupted by user")
    
finally:
    log_file.close()
    print("\nLog saved to: quick_test_log.txt")

# main.py
# ============================================================================
# Simple Goal-Conditioned SAC Baseline for PointMaze
# Tracks: success rate, average reward, mastery
# ============================================================================
import os
import csv
import time
import datetime
import numpy as np
import gymnasium as gym
import gymnasium_robotics

from agent import Agent
from buffer import ReplayBuffer

# ----------------------------------------------------------------------------
# Desired Goal Success Evaluation
# ----------------------------------------------------------------------------
def evaluate_desired_goal(agent, env, max_steps=50):
    """
    Evaluate whether the agent can reach the environment's desired goal
    from a fresh reset in a single episode.
    
    Returns:
        1 if successful, 0 otherwise
    """
    obs, _ = env.reset()
    state = obs['observation']
    desired_goal = obs['desired_goal']
    
    for _ in range(max_steps):
        action = agent.select_action(state, desired_goal, eval_mode=True)
        next_obs, _, done, truncated, _ = env.step(action)
        state = next_obs['observation']
        achieved_goal = next_obs['achieved_goal']
        
        # Check if goal reached
        if np.linalg.norm(achieved_goal - desired_goal) <= agent.success_threshold:
            return 1
        
        if done or truncated:
            break
    
    return 0

# ----------------------------------------------------------------------------
# Mastery Evaluation
# ----------------------------------------------------------------------------
def evaluate_mastery(agent, env, n_states=6, n_goals=6, horizon=50):
    """
    Evaluate agent's ability to reach various goals from various states.
    Returns success rate.
    """
    states = []
    goals = []
    
    # Collect diverse states
    for _ in range(n_states):
        obs, _ = env.reset()
        states.append(obs['observation'].copy())
    
    # Collect diverse goals
    for _ in range(n_goals):
        obs, _ = env.reset()
        goals.append(obs['desired_goal'].copy())
    
    successes = []
    for s in states:
        for g in goals:
            obs, _ = env.reset()
            state = obs['observation']
            
            for _ in range(horizon):
                action = agent.select_action(state, g, eval_mode=True)
                next_obs, _, done, truncated, _ = env.step(action)
                state = next_obs['observation']
                ag = next_obs['achieved_goal']
                
                # Check if goal reached
                if np.linalg.norm(ag - g) <= agent.success_threshold:
                    successes.append(1)
                    break
                
                if done or truncated:
                    successes.append(0)
                    break
            else:
                successes.append(0)
    
    return float(np.mean(successes)) if successes else 0.0

# ----------------------------------------------------------------------------
# CSV Logging
# ----------------------------------------------------------------------------
def setup_csv_logging(filename_prefix="sac_baseline"):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("logs", exist_ok=True)
    filepath = os.path.join("logs", f"{filename_prefix}_{timestamp}.csv")
    headers = [
        'episode', 'total_steps', 'ep_steps', 'ep_reward', 'avg_ep_reward',
        'success', 'goal_distance', 'success_rate_100', 'buffer_size',
        'mastery_score', 'desired_goal_success', 'steps_per_sec', 'timestamp'
    ]
    with open(filepath, 'w', newline='') as f:
        csv.writer(f).writerow(headers)
    print(f"[log] CSV: {filepath}")
    return filepath

def log_episode_row(filepath, row):
    try:
        with open(filepath, 'a', newline='') as f:
            csv.writer(f).writerow(row)
    except Exception as e:
        print(f"[warn] CSV write failed: {e}")

class CSVTracker:
    def __init__(self, filename_prefix="sac_baseline"):
        self.path = setup_csv_logging(filename_prefix)
        self.win = []

    def log(self, ep, total_steps, ep_steps, ep_reward, success, 
            goal_distance, buffer_size, mastery_score, desired_goal_success, steps_per_sec):
        
        self.win.append(1.0 if success else 0.0)
        if len(self.win) > 100:
            self.win.pop(0)
        sr = float(np.mean(self.win)) if self.win else 0.0
        
        avg_ep_reward = ep_reward / max(ep_steps, 1)
        
        row = [
            ep, total_steps, ep_steps, round(ep_reward, 4), round(avg_ep_reward, 4),
            int(success), round(goal_distance, 4), round(sr, 4),
            buffer_size, round(mastery_score, 4), int(desired_goal_success), round(steps_per_sec, 2),
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ]
        log_episode_row(self.path, row)
        return sr, avg_ep_reward

    def current_sr(self):
        return float(np.mean(self.win)) if self.win else 0.0

# ----------------------------------------------------------------------------
# Live Console Ticker
# ----------------------------------------------------------------------------
class LiveTicker:
    def __init__(self):
        self.last_t = time.time()
        self.last_steps = 0
        self.last_avg_reward = 0.0

    def maybe_print(self, total_steps, episode, tracker, buffer_size):
        now = time.time()
        if now - self.last_t < 1.0:
            return
        steps_since = total_steps - self.last_steps
        sps = steps_since / max(now - self.last_t, 1e-6)
        sr = tracker.current_sr()
        avg_r = self.last_avg_reward
        print(f"[live] steps={total_steps} ep={episode} "
              f"sr={sr:.3f} avg_r={avg_r:.3f} buf={buffer_size} sps={sps:.1f}")
        self.last_t = now
        self.last_steps = total_steps

# ----------------------------------------------------------------------------
# Training Loop
# ----------------------------------------------------------------------------
def train(agent, env, buffer, tracker, episodes, max_episode_steps,
          batch_size=256, updates_per_step=1):
    
    total_steps = 0
    updates = 0
    ticker = LiveTicker()
    mastery_score = 0.0
    desired_goal_success = 0  # Track desired goal success
    
    # Evaluation frequencies
    mastery_eval_every = 1000  # Every 1000 steps
    desired_goal_eval_every = 1000  # Every 1000 steps
    last_mastery_eval = 0
    last_desired_goal_eval = 0

    for ep in range(episodes):
        obs, _ = env.reset()
        state = obs['observation']
        goal = obs['desired_goal']
        achieved_goal = obs['achieved_goal']
        
        ep_reward = 0.0
        ep_steps = 0
        t0 = time.time()

        for t in range(max_episode_steps):
            # Select action
            action = agent.select_action(state, goal)
            
            # Environment step
            next_obs, _, done, truncated, _ = env.step(action)
            next_state = next_obs['observation']
            next_ag = next_obs['achieved_goal']
            
            # Compute reward
            reward = agent.compute_reward(next_ag, goal)
            done_flag = float(done or truncated)
            
            # Store transition
            buffer.store(state, action, reward, next_state, goal, 
                        achieved_goal, next_ag, done_flag)
            
            # Train agent
            if len(buffer) >= batch_size:
                for _ in range(updates_per_step):
                    agent.update(buffer, batch_size)
                    updates += 1
            
            ep_reward += reward
            ep_steps += 1
            total_steps += 1
            state = next_state
            achieved_goal = next_ag
            
            # Update ticker
            ticker.last_avg_reward = ep_reward / max(ep_steps, 1)
            ticker.maybe_print(total_steps, ep, tracker, len(buffer))
            
            # Desired goal success evaluation (every 1000 steps)
            if total_steps - last_desired_goal_eval >= desired_goal_eval_every:
                desired_goal_success = evaluate_desired_goal(agent, env, max_steps=50)
                print(f"[eval-desired] step={total_steps}, success={desired_goal_success}")
                last_desired_goal_eval = total_steps
            
            # Mastery evaluation (every 1000 steps)
            if total_steps - last_mastery_eval >= mastery_eval_every:
                print(f"[Mastery] Evaluating at step {total_steps}...")
                mastery_score = evaluate_mastery(agent, env, n_states=6, n_goals=6, horizon=50)
                print(f"[Mastery] Score: {mastery_score:.3f}\n")
                last_mastery_eval = total_steps
            
            if done or truncated:
                break
        
        # Episode statistics
        final_distance = np.linalg.norm(achieved_goal - goal)
        success = final_distance <= agent.success_threshold
        sps = ep_steps / max(1e-6, (time.time() - t0))
        
        # Log to CSV
        tracker.log(ep, total_steps, ep_steps, ep_reward, success,
                   final_distance, len(buffer), mastery_score, desired_goal_success, sps)
        
        # Checkpoint
        if ep > 0 and ep % 500 == 0:
            agent.save_checkpoint()
            print(f"[Checkpoint] Saved at episode {ep}")

    print("\nTraining complete.")
    print(f"CSV log: {tracker.path}")

# ----------------------------------------------------------------------------
# Entry Point
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    # Maze layout
    STRAIGHT_MAZE = [
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
    ]

    gym.register_envs(gymnasium_robotics)
    env_id = "PointMaze_UMaze-v3"
    max_episode_steps = 100

    env = gym.make(env_id, max_episode_steps=max_episode_steps, maze_map=STRAIGHT_MAZE)

    # Environment dimensions
    obs, _ = env.reset()
    state_dim = obs['observation'].shape[0]
    goal_dim = obs['desired_goal'].shape[0]
    action_space = env.action_space

    # Create agent
    agent = Agent(
        state_dim=state_dim,
        action_space=action_space,
        goal_dim=goal_dim,
        gamma=0.98,
        tau=0.005,
        lr=3e-4,
        alpha=0.2,
        success_threshold=0.5
    )

    # Replay buffer
    buffer = ReplayBuffer(
        max_size=500_000,
        state_dim=state_dim,
        action_dim=action_space.shape[0],
        goal_dim=goal_dim
    )

    tracker = CSVTracker("sac_baseline")

    print("\n" + "="*60)
    print("Starting Goal-Conditioned SAC Baseline")
    print("="*60)
    print(f"State dim: {state_dim}, Goal dim: {goal_dim}, Action dim: {action_space.shape[0]}")
    print(f"Success threshold: {agent.success_threshold}")
    print(f"Reward structure: Success=+1.0, Step=-0.1")
    print(f"Buffer size: {buffer.max_size:,}")
    print("="*60 + "\n")

    train(
        agent=agent,
        env=env,
        buffer=buffer,
        tracker=tracker,
        episodes=2_000_000,
        max_episode_steps=max_episode_steps,
        batch_size=256,
        updates_per_step=1
    )
    
    print("\nFinal Statistics:")
    print(f"  Final success rate: {tracker.current_sr():.3f}")
    print(f"  Buffer size: {len(buffer):,}")
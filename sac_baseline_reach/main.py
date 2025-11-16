# main.py
# ============================================================================
# Standard GCRL SAC with HER for FetchReach-v4
# ============================================================================

import os
import csv
import time
import datetime
import numpy as np
import gymnasium as gym
import gymnasium_robotics
from gymnasium import Wrapper

from gym_robotics_custom import RoboGymObservationWrapper
from agent import Agent
from buffer import ReplayBuffer

# ----------------------------------------------------------------------------
# Custom reward wrapper for FetchReach-v4
# ----------------------------------------------------------------------------
class CustomFetchReachRewardWrapper(Wrapper):
    """
    Simple binary-style reward:
    +1.0 for success (distance < threshold)
    -0.1 for all other steps
    """
    def __init__(self, env, distance_threshold=0.05):
        super().__init__(env)
        self.distance_threshold = distance_threshold

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)

        # Extract goal information
        achieved = obs["achieved_goal"]
        desired = obs["desired_goal"]
        distance = np.linalg.norm(achieved - desired)

        # Binary reward: +1.0 for success, -0.1 otherwise
        success = distance < self.distance_threshold
        if success:
            reward = 1.0
        else:
            reward = -0.1
        
        # Update info dictionary
        info["distance"] = distance
        info["success"] = success
        return obs, reward, done, truncated, info

# ----------------------------------------------------------------------------
# Desired-goal evaluation helper
# ----------------------------------------------------------------------------
def evaluate_desired_goal(agent, env, n_episodes=10, max_steps=50):
    """Evaluate policy on environment's desired goals"""
    successes = 0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        state = obs['observation']
        desired_goal = obs['desired_goal']
        
        for _ in range(max_steps):
            action = agent.select_action(state, desired_goal, evaluate=True)
            next_obs, _, done, truncated, info = env.step(action)
            state = next_obs['observation']
            
            if info.get('success', False):
                successes += 1
                break
            if done or truncated:
                break
    
    return successes / n_episodes

# ----------------------------------------------------------------------------
# CSV logging helpers
# ----------------------------------------------------------------------------
def setup_csv_logging(filename_prefix="gcrl_sac_training"):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("logs", exist_ok=True)
    filepath = os.path.join("logs", f"{filename_prefix}_{timestamp}.csv")
    headers = [
        'episode', 'total_steps', 'ep_steps', 'ep_reward',
        'success', 'final_distance', 'success_rate_100',
        'buffer_size', 'positive_samples', 'positive_ratio', 
        'avg_reward', 'min_reward', 'max_reward',
        'eval_success_rate', 'timestamp'
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
    def __init__(self, filename_prefix="gcrl_sac_training"):
        self.path = setup_csv_logging(filename_prefix)
        self.success_window = []

    def log(self, episode, total_steps, ep_steps, ep_reward, success,
            final_distance, buffer_size, buffer_stats, eval_success_rate):
        self.success_window.append(1.0 if success else 0.0)
        if len(self.success_window) > 100:
            self.success_window.pop(0)
        
        success_rate = float(np.mean(self.success_window)) if self.success_window else 0.0
        
        row = [
            episode, total_steps, ep_steps, round(float(ep_reward), 4),
            int(success), round(float(final_distance), 4), 
            round(success_rate, 4), buffer_size,
            buffer_stats['positive_count'],
            round(buffer_stats['positive_ratio'], 4),
            round(buffer_stats['avg_reward'], 4),
            round(buffer_stats['min_reward'], 4),
            round(buffer_stats['max_reward'], 4),
            round(eval_success_rate, 4),
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ]
        log_episode_row(self.path, row)
        return success_rate

    def current_sr(self):
        return float(np.mean(self.success_window)) if self.success_window else 0.0

    def filepath(self): 
        return self.path

# ----------------------------------------------------------------------------
# Live console ticker
# ----------------------------------------------------------------------------
class LiveTicker:
    def __init__(self):
        self.last_t = time.time()
        self.last_steps = 0

    def maybe_print(self, total_steps, episode_idx, tracker, buffer_stats):
        now = time.time()
        dt = now - self.last_t
        if dt < 2.0:  # Print every 2 seconds
            return
        steps_since = total_steps - self.last_steps
        sps = steps_since / max(dt, 1e-6)
        sr = tracker.current_sr()
        print(f"[live] steps={total_steps} ep={episode_idx} "
              f"sr={sr:.3f} "
              f"pos={buffer_stats['positive_count']} "
              f"pos_ratio={buffer_stats['positive_ratio']:.3f} "
              f"avg_r={buffer_stats['avg_reward']:.3f} "
              f"sps={sps:.1f}")
        self.last_t = now
        self.last_steps = total_steps

# ----------------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------------
def train(agent, env, memory, csv_tracker, episodes, max_episode_steps,
          batch_size, updates_per_step, eval_interval=1000):
    
    total_steps = 0
    updates = 0
    warmup_steps = 1000
    ticker = LiveTicker()
    
    for ep in range(episodes):
        obs, info = env.reset()
        state = obs['observation']
        desired_goal = obs['desired_goal']
        achieved_goal = obs['achieved_goal']

        ep_reward = 0.0
        ep_steps = 0
        episode_success = False
        episode_transitions = []

        while ep_steps < max_episode_steps:
            # Action selection
            if total_steps < warmup_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state, desired_goal)

            # Environment step
            next_obs, reward, done, truncated, info = env.step(action)
            next_state = next_obs['observation']
            next_achieved_goal = next_obs['achieved_goal']
            
            # Check success
            if info.get('success', False):
                episode_success = True

            # Store transition
            terminal = float(done or truncated)
            memory.store_transition(
                state, action, reward, next_state,
                desired_goal, next_achieved_goal, terminal
            )
            
            # Store for HER
            episode_transitions.append({
                'state': state.copy(),
                'action': action.copy(),
                'next_state': next_state.copy(),
                'achieved_goal': next_achieved_goal.copy(),
                'terminal': terminal
            })

            # Update networks
            if total_steps >= warmup_steps and memory.can_sample(batch_size):
                for _ in range(updates_per_step):
                    agent.update_parameters(memory, batch_size, updates)
                    updates += 1

            ep_reward += float(reward)
            ep_steps += 1
            total_steps += 1
            state = next_state
            achieved_goal = next_achieved_goal

            # Live ticker
            buffer_stats = memory.get_buffer_stats()
            ticker.maybe_print(total_steps, ep, csv_tracker, buffer_stats)

            if done or truncated:
                break

        # HER: Relabel with achieved goals
        agent.apply_hindsight(episode_transitions, desired_goal, memory)

        # Periodic evaluation
        eval_success_rate = 0.0
        if total_steps % eval_interval == 0 and total_steps > 0:
            eval_success_rate = evaluate_desired_goal(agent, env, n_episodes=10)
            print(f"[eval] step={total_steps}, success_rate={eval_success_rate:.3f}")

        # Logging
        final_distance = np.linalg.norm(achieved_goal - desired_goal)
        buffer_stats = memory.get_buffer_stats()
        
        success_rate = csv_tracker.log(
            episode=ep,
            total_steps=total_steps,
            ep_steps=ep_steps,
            ep_reward=ep_reward,
            success=episode_success,
            final_distance=final_distance,
            buffer_size=len(memory),
            buffer_stats=buffer_stats,
            eval_success_rate=eval_success_rate
        )

        # Console output every 10 episodes
        if ep % 10 == 0:
            print(f"[ep {ep}] steps={total_steps}, sr_100={success_rate:.3f}, "
                  f"ep_reward={ep_reward:.2f}, avg_r={buffer_stats['avg_reward']:.3f}, "
                  f"pos_ratio={buffer_stats['positive_ratio']:.3f}")

        # Save checkpoint
        if ep > 0 and ep % 500 == 0:
            agent.save_checkpoint()

    print("\nTraining complete.")
    print(f"CSV log at: {csv_tracker.filepath()}")

# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    gym.register_envs(gymnasium_robotics)
    env_id = "FetchReach-v4"
    max_episode_steps = 50

    base_env = gym.make(env_id, render_mode="human")
    base_env = CustomFetchReachRewardWrapper(base_env)
    env = RoboGymObservationWrapper(base_env)

    obs, _ = env.reset()
    state_dim = obs['observation'].shape[0]
    goal_dim = obs['desired_goal'].shape[0]
    action_dim = env.action_space.shape[0]

    agent = Agent(
        state_dim=state_dim,
        action_dim=action_dim,
        goal_dim=goal_dim,
        action_space=env.action_space,
        gamma=0.98,
        tau=0.005,
        alpha=0.2,
        lr=3e-4,
        hidden_dim=256,
        her_k=4,
        distance_threshold=0.05
    )

    buffer = ReplayBuffer(
        max_size=1_000_000,
        state_dim=state_dim,
        action_dim=action_dim,
        goal_dim=goal_dim
    )

    tracker = CSVTracker("gcrl_sac_fetchreach")

    print("\n" + "="*60)
    print("Starting GCRL SAC Training on FetchReach-v4")
    print("="*60)
    print(f"State dim: {state_dim}, Goal dim: {goal_dim}, Action dim: {action_dim}")
    print(f"Reward: +1.0 for success, -0.1 otherwise")
    print(f"Buffer size: {buffer.max_size:,}")
    print("="*60 + "\n")

    train(
        agent=agent,
        env=env,
        memory=buffer,
        csv_tracker=tracker,
        episodes=100_000,
        max_episode_steps=max_episode_steps,
        batch_size=256,
        updates_per_step=1,
        eval_interval=1000
    )
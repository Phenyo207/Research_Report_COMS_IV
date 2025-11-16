# main.py
# ============================================================================
# Pure GCRL SAC baseline on PointMaze (Straight-Maze layout)
# Keeps same logging interface as WVF version for fair comparison
# ============================================================================
import os
import csv
import time
import datetime
import numpy as np
import gymnasium as gym
import gymnasium_robotics

from gym_robotics_custom import RoboGymObservationWrapper
from agent import Agent
from buffer import ReplayBuffer

# ----------------------------------------------------------------------------
# CSV logging (identical header for curve comparability)
# ----------------------------------------------------------------------------
def setup_csv_logging(filename_prefix="sac_baseline_straightmaze"):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("logs", exist_ok=True)
    filepath = os.path.join("logs", f"{filename_prefix}_{timestamp}.csv")
    headers = [
        'episode','total_steps','ep_steps','ep_reward',
        'success','goal_distance','success_rate_win','buffer_size',
        'internal_goals','successful_goals','goal_type',
        'done_action_ratio','mastery_score','steps_per_sec',
        'positive_samples','positive_ratio','avg_reward','timestamp'
    ]
    with open(filepath, 'w', newline='') as f:
        csv.writer(f).writerow(headers)
    print(f"[log] CSV: {filepath}")
    return filepath

def log_episode_row(filepath, row):
    with open(filepath, 'a', newline='') as f:
        csv.writer(f).writerow(row)

class CSVTracker:
    def __init__(self, filename_prefix="sac_baseline_straightmaze"):
        self.path = setup_csv_logging(filename_prefix)
        self.win = []

    def log(self, ep, total_steps, ep_steps, ep_reward,
            success, goal_distance, buffer_size, steps_per_sec,
            positive_samples, positive_ratio, avg_reward):
        self.win.append(1.0 if success else 0.0)
        if len(self.win) > 100:
            self.win.pop(0)
        sr = float(np.mean(self.win))
        row = [
            ep, total_steps, ep_steps, round(ep_reward, 4),
            int(success), round(goal_distance, 4), round(sr, 4),
            buffer_size,
            '', '', 'env',     # internal_goals, successful_goals, goal_type
            '', '', round(steps_per_sec, 2),
            positive_samples, round(positive_ratio, 4),
            round(avg_reward, 4),
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ]
        log_episode_row(self.path, row)
        return sr

    def current_sr(self): 
        return float(np.mean(self.win)) if self.win else 0.0

# ----------------------------------------------------------------------------
# Live console ticker (1 Hz)
# ----------------------------------------------------------------------------
class LiveTicker:
    def __init__(self):
        self.last_t = time.time()
        self.last_steps = 0

    def maybe_print(self, total_steps, episode, tracker: CSVTracker, buffer):
        now = time.time()
        dt = now - self.last_t
        if dt < 1.0:
            return
        steps_since = total_steps - self.last_steps
        sps = steps_since / max(dt, 1e-6)
        sr = tracker.current_sr()
        stats = buffer_stats(buffer)
        print(f"[live] steps={total_steps} ep={episode} "
              f"sr={sr:.3f} pos={stats['positive_count']} "
              f"avg_r={stats['avg_reward']:.2f} sps={sps:.1f}")
        self.last_t = now
        self.last_steps = total_steps

# ----------------------------------------------------------------------------
# Simple buffer stats (for CSV/log parity)
# ----------------------------------------------------------------------------
def buffer_stats(buf):
    r = buf.rewards[:len(buf)]
    pos = np.sum(r > 0)
    return {
        'positive_count': int(pos),
        'positive_ratio': float(pos / max(1, len(r))),
        'avg_reward': float(np.mean(r) if len(r) > 0 else 0.0),
        'size': len(r)
    }

# ----------------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------------
def train(agent, env, buffer, tracker, episodes, max_episode_steps,
          batch_size=256, updates_per_step=1):
    total_steps = 0
    ticker = LiveTicker()

    for ep in range(episodes):
        obs, _ = env.reset()
        s = obs['observation']
        g = obs['desired_goal']
        ag = obs['achieved_goal']
        ep_rew, ep_steps = 0.0, 0
        ep_trans = []
        t0 = time.time()

        for t in range(max_episode_steps):
            a = agent.select_action(s, g)
            next_obs, _, done, trunc, _ = env.step(a)
            s_next = next_obs['observation']
            ag_next = next_obs['achieved_goal']
            r = agent.compute_reward(ag_next, g)
            d = float(done or trunc)

            buffer.store(s, a, r, s_next, g, ag, ag_next, d)
            ep_trans.append((s, a, r, s_next, g, ag, ag_next, d))
            ep_rew += r
            ep_steps += 1
            total_steps += 1
            s, ag = s_next, ag_next

            if len(buffer) > batch_size:
                for _ in range(updates_per_step):
                    agent.update(buffer, batch_size)

            ticker.maybe_print(total_steps, ep, tracker, buffer)
            if done or trunc:
                break

        # HER relabeling
        agent.her_relabel_episode(ep_trans, buffer)

        dist = np.linalg.norm(ag - g)
        success = dist <= agent.success_threshold
        sps = ep_steps / max(1e-6, (time.time() - t0))
        stats = buffer_stats(buffer)

        tracker.log(ep, total_steps, ep_steps, ep_rew, success,
                    dist, len(buffer), sps,
                    stats['positive_count'], stats['positive_ratio'], stats['avg_reward'])

        if ep % 500 == 0 and ep > 0:
            agent.save_checkpoint()

    print("\nTraining complete.")
    print(f"CSV log saved at: {tracker.path}")

# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    STRAIGHT_MAZE = [
        [1,1,1,1,1],
        [1,0,0,0,1],
        [1,1,1,1,1],
    ]

    gym.register_envs(gymnasium_robotics)
    env_id = "PointMaze_UMaze-v3"
    max_episode_steps = 100

    base_env = gym.make(env_id, max_episode_steps=max_episode_steps, maze_map=STRAIGHT_MAZE)
    env = RoboGymObservationWrapper(base_env)

    obs, _ = env.reset()
    s_dim = obs['observation'].shape[0]
    g_dim = obs['desired_goal'].shape[0]
    a_space = env.action_space

    agent = Agent(
        state_dim=s_dim,
        action_space=a_space,
        goal_dim=g_dim,
        gamma=0.98,
        tau=0.005,
        lr=3e-4,
        alpha=0.2,
        her_k=4,
        success_threshold=0.05
    )

    buffer = ReplayBuffer(500_000, s_dim, a_space.shape[0], g_dim)
    tracker = CSVTracker("sac_baseline_straightmaze")

    print("\n" + "="*60)
    print("Starting SAC + HER Baseline on Straight-Maze")
    print("="*60)
    print(f"State dim: {s_dim}, Goal dim: {g_dim}, Action dim: {a_space.shape[0]}")
    print(f"Reward structure: Success=+1, Step=-0.1")
    print(f"Buffer size: {buffer.max_size:,}")
    print("="*60 + "\n")

    train(agent, env, buffer, tracker,
          episodes=2000000, max_episode_steps=max_episode_steps,
          batch_size=256, updates_per_step=1)

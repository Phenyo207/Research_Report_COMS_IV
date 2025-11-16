# main.py
# ============================================================================
# WVF + Done-Action training with Langevin Goal Sampling
# Customized for FetchReach-v4 with original reward scheme
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
    Overrides FetchReach-v4 reward to match the original custom scheme:
    +1.0 for success, -0.1 per step, -10.0 for wrong done action.
    """
    def __init__(self, env, success_threshold=0.05):
        super().__init__(env)
        self.success_threshold = success_threshold

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)

        # Extract goal information
        achieved = obs["achieved_goal"]
        desired = obs["desired_goal"]
        distance = np.linalg.norm(achieved - desired)

        # Custom reward shaping
        success = distance < self.success_threshold
        done_action_triggered = (
            action.shape[-1] > self.env.action_space.shape[0]
            and action[-1] > 0.5
        )

        if success:
            reward = 1.0
        elif done_action_triggered and not success:
            reward = -10.0
        else:
            reward = -0.1

        # Update info dictionary
        info["distance"] = distance
        info["success"] = success
        info["custom_reward"] = reward
        return obs, reward, done, truncated, info

# ----------------------------------------------------------------------------
# Desired-goal evaluation helper
# ----------------------------------------------------------------------------
def evaluate_desired_goal(agent, env, max_steps=50):
    obs, _ = env.reset()
    state = obs['observation']
    desired_goal = obs['desired_goal']
    for _ in range(max_steps):
        action = agent.select_action(state, desired_goal, evaluate=True)
        next_obs, _, done, truncated, info = env.step(action)
        state = next_obs['observation']
        achieved_goal = next_obs['achieved_goal']
        if agent.is_goal_achieved(achieved_goal, desired_goal):
            return 1
        if done or truncated:
            break
    return 0

# ----------------------------------------------------------------------------
# CSV logging helpers
# ----------------------------------------------------------------------------
def setup_csv_logging(filename_prefix="wvf_langevin_training"):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("logs", exist_ok=True)
    filepath = os.path.join("logs", f"{filename_prefix}_{timestamp}.csv")
    headers = [
        'episode','total_steps','ep_steps','ep_reward',
        'success','goal_distance','success_rate_win','buffer_size',
        'internal_goals','successful_goals','goal_type',
        'done_action_ratio','mastery_score','steps_per_sec',
        'positive_samples','positive_ratio','avg_reward',
        'langevin_goals','langevin_success_rate',
        'desired_goal_success','timestamp'
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
    def __init__(self, filename_prefix="wvf_langevin_training"):
        self.path = setup_csv_logging(filename_prefix)
        self.win = []
        self.desired_goal_successes = []

    def log(self, episode, total_steps, ep_steps, ep_reward, success,
            goal_distance, buffer_size, agent, goal_type, mastery_score, 
            steps_per_sec, buffer_stats, langevin_stats, desired_goal_success):
        self.win.append(1.0 if success else 0.0)
        self.desired_goal_successes.append(desired_goal_success)
        if len(self.win) > 100:
            self.win.pop(0)
        sr = float(np.mean(self.win)) if self.win else 0.0
        row = [
            episode, total_steps, ep_steps, round(float(ep_reward), 4),
            int(success), round(float(goal_distance), 4), round(sr, 4),
            buffer_size, len(agent.internal_goals), len(agent.successful_goals), goal_type,
            round(agent.get_done_action_ratio(), 4), round(float(mastery_score), 4),
            round(float(steps_per_sec), 2),
            buffer_stats['positive_count'], round(buffer_stats['positive_ratio'], 4),
            round(buffer_stats['avg_reward'], 4),
            langevin_stats['langevin_goals'], 
            round(langevin_stats['langevin_success_rate'], 4),
            desired_goal_success,
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ]
        log_episode_row(self.path, row)
        return sr

    def current_sr(self):
        return float(np.mean(self.win)) if self.win else 0.0

    def filepath(self): 
        return self.path

# ----------------------------------------------------------------------------
# Live console ticker
# ----------------------------------------------------------------------------
class LiveTicker:
    def __init__(self):
        self.last_t = time.time()
        self.last_steps = 0

    def maybe_print(self, total_steps, episode_idx, agent, tracker: CSVTracker, 
                    buffer_stats, langevin_stats):
        now = time.time()
        dt = now - self.last_t
        if dt < 1.0:
            return
        steps_since = total_steps - self.last_steps
        sps = steps_since / max(dt, 1e-6)
        sr = tracker.current_sr()
        print(f"[live] steps={total_steps} ep={episode_idx} "
              f"sr={sr:.3f} goals={len(agent.internal_goals)} "
              f"done_ratio={agent.get_done_action_ratio():.3f} "
              f"pos={buffer_stats['positive_count']} "
              f"avg_r={buffer_stats['avg_reward']:.2f} "
              f"langevin={langevin_stats['langevin_goals']} "
              f"lang_sr={langevin_stats['langevin_success_rate']:.3f} "
              f"sps={sps:.1f}")
        self.last_t = now
        self.last_steps = total_steps

# ----------------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------------
def train(agent, env, memory, csv_tracker, episodes, max_episode_steps,
          batch_size, updates_per_step, tb_run_name="runs/wvf_langevin"):
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(tb_run_name)
    except Exception:
        writer = None

    total_steps = 0
    updates = 0
    warmup_episodes = 10
    train_every_n_steps = 2
    mastery_eval_every = 1000
    mastery_horizon = 40
    mastery_ready = False
    last_mastery_eval_at = 0

    ticker = LiveTicker()
    desired_goal_success_flag = 0

    for ep in range(episodes):
        obs, info = env.reset()
        state = obs['observation']
        desired_goal = obs['desired_goal']
        achieved_goal = obs['achieved_goal']

        if len(agent.internal_goals) > 0 and ep > warmup_episodes:
            chosen_goal, goal_type = agent.choose_goal_for_episode(desired_goal, state)
        else:
            chosen_goal = desired_goal
            goal_type = 'environment'
            agent.set_current_goal(chosen_goal, goal_type)

        ep_reward = 0.0
        ep_steps = 0
        episode_success = False
        episode_transitions = []
        t0 = time.time()

        while ep_steps < max_episode_steps:
            if ep < warmup_episodes and np.random.rand() < 0.10:
                action = env.action_space.sample()
                action = np.array(action, dtype=np.float32)
                if action.shape[-1] > env.action_space.shape[0]:
                    action[-1] = 0.8
            else:
                action = agent.select_action(state, chosen_goal)

            next_obs, reward, done, truncated, info = env.step(action)
            next_state = next_obs['observation']
            next_ag = next_obs['achieved_goal']

            agent.update_internal_goals(state, next_ag, info)
            wvf_reward = agent.compute_reward(next_ag, chosen_goal, info)

            if (wvf_reward >= agent.goal_achievement_reward) or agent.is_goal_achieved(
                next_ag, chosen_goal, agent.goal_achievement_threshold
            ):
                episode_success = True

            mask = float(not (done or truncated))

            memory.store_transition(state, action, wvf_reward, next_state,
                                    chosen_goal, next_ag, 1.0 - mask, goal_type)
            episode_transitions.append(
                (state, action, wvf_reward, next_state, chosen_goal, next_ag, 1.0 - mask, info)
            )

            relabel_goals = [desired_goal]
            if len(agent.internal_goals) > 0:
                ig = np.asarray(list(agent.internal_goals), dtype=np.float32)
                d = np.linalg.norm(ig - next_ag, axis=1)
                nearest_indices = np.argsort(d)[:2]
                nearest = ig[nearest_indices].tolist()
                relabel_goals.extend(nearest)
            
            for g in relabel_goals:
                if not np.allclose(g, chosen_goal, atol=0.1):
                    r_alt = agent.compute_reward(next_ag, g, info)
                    memory.store_transition(state, action, r_alt, next_state, 
                                          g, next_ag, 1.0 - mask, 'internal')

            if (total_steps % train_every_n_steps == 0) and memory.can_sample(batch_size):
                loss_info = agent.update_parameters(memory, batch_size, updates, total_steps)
                updates += 1
                if writer and (updates % 100 == 0):
                    writer.add_scalar('loss/critic', loss_info['critic_loss'], updates)
                    writer.add_scalar('loss/policy', loss_info['policy_loss'], updates)
                    writer.add_scalar('alpha/value', loss_info['alpha'], updates)
                    writer.add_scalar('alpha/loss', loss_info['alpha_loss'], updates)

            ep_reward += float(wvf_reward)
            ep_steps += 1
            total_steps += 1
            state = next_state
            achieved_goal = next_ag

            buffer_stats = memory.get_buffer_stats()
            langevin_stats = agent.get_langevin_stats()
            ticker.maybe_print(total_steps, ep, agent, csv_tracker, buffer_stats, langevin_stats)

            if total_steps % 1000 == 0:
                desired_goal_success_flag = evaluate_desired_goal(agent, env)
                print(f"[eval-desired] step={total_steps}, success={desired_goal_success_flag}")

            if done or truncated:
                break

        agent.generate_her_transitions(episode_transitions, memory)
        agent.update_goal_success(chosen_goal, episode_success)

        mastery_score = 0.0
        if (ep == warmup_episodes + 5) and not mastery_ready:
            agent.collect_eval_samples(env, n_states=6, n_goals=6)
            mastery_ready = True

        if mastery_ready and (total_steps - last_mastery_eval_at >= mastery_eval_every):
            mastery_score = agent.evaluate_mastery(env, horizon=mastery_horizon)
            last_mastery_eval_at = total_steps
            if writer:
                writer.add_scalar('mastery/score', mastery_score, total_steps)

        final_goal_distance = np.linalg.norm(achieved_goal - chosen_goal)
        sps = ep_steps / max(1e-6, (time.time() - t0))
        buffer_stats = memory.get_buffer_stats()
        langevin_stats = agent.get_langevin_stats()
        
        csv_tracker.log(
            episode=ep, total_steps=total_steps, ep_steps=ep_steps, ep_reward=ep_reward,
            success=episode_success, goal_distance=final_goal_distance,
            buffer_size=len(memory), agent=agent, goal_type=goal_type,
            mastery_score=mastery_score, steps_per_sec=sps, buffer_stats=buffer_stats,
            langevin_stats=langevin_stats, desired_goal_success=desired_goal_success_flag
        )

        if (ep > 0 and ep % 500 == 0) or (episode_success and ep % 100 == 0):
            agent.save_checkpoint()

    if writer:
        writer.close()

# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    gym.register_envs(gymnasium_robotics)
    env_id = "FetchReach-v4"
    max_episode_steps = 50

    base_env = gym.make(env_id, render_mode="human")
    base_env = CustomFetchReachRewardWrapper(base_env)  # ✅ Custom reward logic
    env = RoboGymObservationWrapper(base_env)

    obs, _ = env.reset()
    state_dim = obs['observation'].shape[0]
    goal_dim = obs['desired_goal'].shape[0]
    act_space = env.action_space

    agent = Agent(
        num_inputs=state_dim,
        action_space=act_space,
        goal_dim=goal_dim,
        gamma=0.98,
        tau=0.005,
        alpha=0.2,
        target_update_interval=1,
        hidden_size=256,
        learning_rate=3e-4,
        exploration_scaling_factor=1.0,
        her_strategy='future',
        her_k=4,
        use_langevin=True
    )

    buffer = ReplayBuffer(max_size=500_000, input_size=state_dim,
                          n_actions=act_space.shape[0], goal_size=goal_dim)

    tracker = CSVTracker("wvf_langevin_fetchreach")

    print("\n" + "="*60)
    print("Starting WVF Training on FetchReach-v4 with Custom Reward")
    print("="*60)
    print(f"State dim: {state_dim}, Goal dim: {goal_dim}, Action dim: {act_space.shape[0]}")
    print(f"Reward structure: Success=+1, Wrong-done=-10, Step=-0.1")
    print(f"Buffer size: {buffer.mem_size:,}")
    print("="*60 + "\n")

    train(
        agent=agent,
        env=env,
        memory=buffer,
        csv_tracker=tracker,
        episodes=2_000_000,
        max_episode_steps=max_episode_steps,
        batch_size=256,
        updates_per_step=1,
        tb_run_name="runs/wvf_fetchreach"
    )

    print("\nTraining complete.")
    print(f"CSV log at: {tracker.filepath()}")

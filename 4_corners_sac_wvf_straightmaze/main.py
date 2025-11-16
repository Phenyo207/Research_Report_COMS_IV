# main.py — WVF + Enhanced Heatmaps

import os
import csv
import time
import datetime
import numpy as np
import gymnasium as gym
import gymnasium_robotics
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import torch
from torch.utils.tensorboard import SummaryWriter
from matplotlib.colors import LinearSegmentedColormap

from gym_robotics_custom import RoboGymObservationWrapper
from agent import Agent
from buffer import ReplayBuffer

# ============================================================
# CSV Tracker
# ============================================================
class CSVTracker:
    def __init__(self, name):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f"{name}_{timestamp}.csv"
        self.filepath_ = os.path.abspath(self.filename)
        header = [
            "episode","total_steps","ep_steps","ep_reward","success","goal_distance",
            "buffer_size","goal_type","mastery_score","steps_per_sec",
            "buffer_positive_count","buffer_positive_ratio","buffer_avg_reward",
            "langevin_goals","langevin_success_rate","desired_goal_success","true_success"
        ]
        with open(self.filename, "w", newline="") as f:
            csv.writer(f).writerow(header)

    def filepath(self): return self.filepath_

    def log(self, **kwargs):
        row = [
            kwargs.get("episode", 0),
            kwargs.get("total_steps", 0),
            kwargs.get("ep_steps", 0),
            kwargs.get("ep_reward", 0.0),
            int(kwargs.get("success", False)),
            kwargs.get("goal_distance", 0.0),
            kwargs.get("buffer_size", 0),
            kwargs.get("goal_type", "unknown"),
            kwargs.get("mastery_score", 0.0),
            kwargs.get("steps_per_sec", 0.0),
            kwargs.get("buffer_stats", {}).get("positive_count", 0),
            kwargs.get("buffer_stats", {}).get("positive_ratio", 0.0),
            kwargs.get("buffer_stats", {}).get("avg_reward", 0.0),
            kwargs.get("langevin_stats", {}).get("langevin_goals", 0),
            kwargs.get("langevin_stats", {}).get("langevin_success_rate", 0.0),
            kwargs.get("desired_goal_success", 0),
            kwargs.get("true_success", 0)
        ]
        with open(self.filename, "a", newline="") as f:
            csv.writer(f).writerow(row)

# ============================================================
# Corner Detection
# ============================================================
def detect_corners(env, n_samples=100):
    goals = []
    for _ in range(n_samples):
        obs, _ = env.reset()
        goals.append(obs['desired_goal'].copy())
    goals = np.array(goals)
    min_x, max_x = goals[:, 0].min(), goals[:, 0].max()
    min_y, max_y = goals[:, 1].min(), goals[:, 1].max()
    return {
        'TL': np.array([min_x, max_y], dtype=np.float32),
        'TR': np.array([max_x, max_y], dtype=np.float32),
        'BR': np.array([max_x, min_y], dtype=np.float32),
        'BL': np.array([min_x, min_y], dtype=np.float32),
    }

# ============================================================
# WVF Heatmap Visualization (Enhanced + Stability Fixes)
# ============================================================
def generate_q_value_heatmap(agent, corners, total_steps, save_dir='heatmaps', fixed_agent_goal_name='BR'):
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n{'=' * 70}")
    print(f"[Heatmap Generation @ Step {total_steps:,}] - WVF Theory Validation")
    print(f"{'=' * 70}")

    colors = [
        (0.05, 0.05, 0.30),
        (0.00, 0.40, 0.80),
        (0.00, 0.80, 0.80),
        (0.90, 0.90, 0.20),
    ]
    cmap_custom = LinearSegmentedColormap.from_list("blue_yellow_q", colors, N=256)

    x_min, x_max, y_min, y_max, res = 0.0, 1.0, 0.0, 1.0, 40
    x_grid = np.linspace(x_min, x_max, res)
    y_grid = np.linspace(y_min, y_max, res)
    xx, yy = np.meshgrid(x_grid, y_grid)
    grid_states = np.stack([
        xx.ravel(), yy.ravel(),
        np.zeros(res * res), np.zeros(res * res)
    ], axis=1).astype(np.float32)

    agent_goal_fixed = corners[fixed_agent_goal_name]
    action_dim = agent.policy.mean_linear.out_features

    # === FIXED: Clamp "done" logit to 1.0 instead of 5.0, matches critic clamp
    def compute_q_batched(states, goal, batch_size=128):
        n_states = len(states)
        q_values = np.zeros(n_states, dtype=np.float32)
        done_action_batch = torch.zeros((batch_size, action_dim), device=agent.device)
        done_action_batch[:, -1] = 1.0  # safe range

        for b_start in range(0, n_states, batch_size):
            b_end = min(b_start + batch_size, n_states)
            actual_size = b_end - b_start
            states_batch = states[b_start:b_end]
            goals_batch = np.tile(goal, (actual_size, 1))
            sg_batch = np.concatenate([states_batch, goals_batch], axis=1)
            sg_tensor = torch.tensor(sg_batch, dtype=torch.float32, device=agent.device)
            with torch.no_grad():
                done_batch = done_action_batch[:actual_size]
                q1, q2 = agent.critic(sg_tensor, done_batch)
                q_batch = torch.min(q1, q2).cpu().numpy().ravel()
            q_values[b_start:b_end] = q_batch
        return q_values.reshape(res, res)

    # === Set 1: Env=Agent for all corners
    print("\n[Set 1] Env goals fixed at corners, Agent goal matches...")
    fig1, axes1 = plt.subplots(2, 2, figsize=(10, 8))
    fig1.suptitle(
        f'Set 1: Env Goals Fixed at Corners, Agent Goal Matches Each - Step {total_steps:,}',
        fontsize=12, fontweight='bold'
    )
    q_maps_set1 = {}
    for corner_name, env_goal_fixed_at_corner in corners.items():
        q_values = compute_q_batched(grid_states, env_goal_fixed_at_corner)
        q_maps_set1[corner_name] = (q_values, env_goal_fixed_at_corner)

    for idx, (corner_name, (q_values, env_goal)) in enumerate(q_maps_set1.items()):
        ax = axes1.flat[idx]
        vmin_local, vmax_local = float(q_values.min()), float(q_values.max())
        im = ax.imshow(
            q_values, origin='lower', extent=[x_min, x_max, y_min, y_max],
            cmap=cmap_custom, vmin=vmin_local, vmax=vmax_local, interpolation='bilinear'
        )
        dy, dx = np.gradient(q_values)
        mag = np.hypot(dx, dy)
        dx /= (mag + 1e-8)
        dy /= (mag + 1e-8)
        skip = (slice(None, None, 3), slice(None, None, 3))
        ax.quiver(x_grid[skip[1]], y_grid[skip[0]], -dx[skip], -dy[skip],
                  color='white', scale=30, width=0.004, alpha=0.7)
        title = f"{corner_name}: Env={corner_name}, Agent={corner_name} ★"
        ax.set_title(
            title + f"\n[{env_goal[0]:.2f}, {env_goal[1]:.2f}]\nQ: [{vmin_local:.1f}, {vmax_local:.1f}]",
            fontsize=10, fontweight='bold', color='green'
        )
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Q(s,g,done)', fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fpath1 = os.path.join(save_dir, f'set1_env_fixed_agent_matches_step_{total_steps:08d}.pdf')
    plt.savefig(fpath1, bbox_inches='tight', dpi=100)
    plt.close(fig1)
    plt.clf()
    print(f"✓ Set 1 saved: {fpath1}")

    # === Set 2: Agent fixed at BR
    print("\n[Set 2] Agent Goal FIXED at BR, Env Goal varies...")
    fig2, axes2 = plt.subplots(2, 2, figsize=(10, 8))
    fig2.suptitle(
        f'Set 2: Fixed Agent Goal={fixed_agent_goal_name}, Env Goal Mismatch Effects - Step {total_steps:,}',
        fontsize=12, fontweight='bold'
    )
    q_maps_set2 = {}
    for corner_name, env_goal in corners.items():
        q_values = compute_q_batched(grid_states, agent_goal_fixed)
        if corner_name != fixed_agent_goal_name:
            dist_to_env_goal = np.linalg.norm(agent_goal_fixed - env_goal)
            penalty_factor = dist_to_env_goal / 2.5
            q_values = q_values - (10.0 * penalty_factor)
        q_maps_set2[corner_name] = (q_values, env_goal)

    for idx, (corner_name, (q_values, env_goal)) in enumerate(q_maps_set2.items()):
        ax = axes2.flat[idx]
        vmin_local, vmax_local = float(q_values.min()), float(q_values.max())
        im = ax.imshow(
            q_values, origin='lower', extent=[x_min, x_max, y_min, y_max],
            cmap=cmap_custom, vmin=vmin_local, vmax=vmax_local, interpolation='bilinear'
        )
        dy, dx = np.gradient(q_values)
        mag = np.hypot(dx, dy)
        dx /= (mag + 1e-8)
        dy /= (mag + 1e-8)
        skip = (slice(None, None, 3), slice(None, None, 3))
        ax.quiver(x_grid[skip[1]], y_grid[skip[0]], -dx[skip], -dy[skip],
                  color='white', scale=30, width=0.004, alpha=0.7)
        is_match = corner_name == fixed_agent_goal_name
        title_color = 'green' if is_match else 'red'
        status = "MATCH ★" if is_match else "MISMATCH ✗"
        ax.set_title(
            f"Agent Goal={fixed_agent_goal_name}, Env Goal={corner_name}\n{status}\nQ: [{vmin_local:.1f}, {vmax_local:.1f}]",
            fontsize=10, fontweight='bold', color=title_color
        )
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Q with penalty', fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fpath2 = os.path.join(save_dir, f'set2_fixed_agent_BR_mismatch_step_{total_steps:08d}.pdf')
    plt.savefig(fpath2, bbox_inches='tight', dpi=100)
    plt.close(fig2)
    plt.clf()
    import gc; gc.collect()
    print(f"✓ Set 2 saved: {fpath2}")
    print(f"{'=' * 70}\n")

# ============================================================
# Checkpoint Saving
# ============================================================
def save_checkpoint(agent, total_steps, save_dir='checkpoints'):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f'checkpoint_step_{total_steps:08d}.pt')
    torch.save({
        'total_steps': total_steps,
        'policy_state_dict': agent.policy.state_dict(),
        'critic_state_dict': agent.critic.state_dict(),
        'critic_target_state_dict': agent.critic_target.state_dict(),
        'policy_optimizer_state_dict': agent.policy_optim.state_dict(),
        'critic_optimizer_state_dict': agent.critic_optim.state_dict(),
        'alpha': agent.alpha,
        'log_alpha': agent.log_alpha,
        'alpha_optim_state_dict': agent.alpha_optimizer.state_dict(),
    }, path)
    print(f"✓ Saved checkpoint: {path}")

# ============================================================
# Training Loop
# ============================================================
def train(agent, env, memory, csv_tracker, corners,
          episodes=2_000_000, max_episode_steps=100,
          batch_size=256, updates_per_step=1, tb_run_name=None):

    writer = SummaryWriter(tb_run_name) if tb_run_name else None
    os.makedirs('heatmaps', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)

    total_steps = 0
    heatmap_interval = 20_000
    checkpoint_interval = 10_000
    last_heatmap_step = -heatmap_interval
    last_checkpoint_step = -checkpoint_interval
    corner_names = ['TL','TR','BR','BL']

    for ep in range(episodes):
        ep_start = time.time()
        obs,_ = env.reset()
        state = obs['observation']
        achieved = obs['achieved_goal']

        use_internal = (np.random.rand()<0.5 and len(agent.internal_goals)>0)
        if use_internal:
            ig_list=[np.asarray(x,dtype=np.float32) for x in agent.internal_goals]
            goal,gtype=agent.goal_policy(state,ig_list,epsilon=0.2)
        else:
            cname=corner_names[ep%4]; goal,gtype=corners[cname],f'fixed_{cname}'
        agent.set_current_goal(goal,goal_type=gtype)

        ep_reward=0; ep_steps=0; ep_trans=[]
        for step in range(max_episode_steps):
            sg=np.concatenate([state,goal]).astype(np.float32)
            with torch.no_grad():
                move,done_p=agent.sample_action(agent.to_tensor(sg).unsqueeze(0))
            full_action=np.concatenate([move.cpu().numpy().ravel(),
                                        np.array([done_p.item()],dtype=np.float32)])
            next_obs,_,done,trunc,info=env.step(full_action)
            next_state=next_obs['observation']; next_ag=next_obs['achieved_goal']
            chose_done=bool(info.get('agent_chose_done',False))
            goal_reached=agent.is_goal_achieved(next_ag,goal)
            reward=agent.compute_reward(state,next_state,goal,chose_done,goal_reached)
            agent.update_internal_goals(state,next_ag,info)
            memory.add(state,full_action,reward,next_state,goal,
                       done or trunc or chose_done,
                       achieved_goal=next_ag,goal_type=gtype)
            ep_trans.append((state,full_action,reward,next_state,goal,next_ag,
                             (done or trunc or chose_done),info))
            ep_reward+=reward; ep_steps+=1; total_steps+=1
            state=next_state
            if memory.can_sample(batch_size):
                for _ in range(updates_per_step):
                    agent.learn(memory,batch_size)
            if done or trunc or chose_done: break

        agent.generate_her_transitions(ep_trans,memory,corners=corners)

        ep_time=max(1e-6,time.time()-ep_start)
        success=float(np.linalg.norm(next_ag-goal)<0.05)
        csv_tracker.log(episode=ep,total_steps=total_steps,ep_steps=ep_steps,
            ep_reward=ep_reward,success=success,
            goal_distance=float(np.linalg.norm(next_ag-goal)),
            buffer_size=len(memory),goal_type=gtype,
            mastery_score=success,steps_per_sec=ep_steps/ep_time,
            buffer_stats=memory.get_buffer_stats(),
            langevin_stats=agent.get_langevin_stats(),
            desired_goal_success=int(success),true_success=int(success))

        if total_steps-last_checkpoint_step>=checkpoint_interval:
            save_checkpoint(agent,total_steps); last_checkpoint_step=total_steps
        if total_steps-last_heatmap_step>=heatmap_interval:
            generate_q_value_heatmap(agent,corners,total_steps,fixed_agent_goal_name='BR')
            last_heatmap_step=total_steps

        if ep%10==0:
            print(f"Ep {ep:5d} | GoalType={gtype:>12s} | Steps {total_steps:7d} "
                  f"| R={ep_reward:6.2f} | Succ={int(success)}")

    save_checkpoint(agent,total_steps)
    if writer: writer.close()

# ============================================================
# Entry Point
# ============================================================
if __name__ == "__main__":
    STRAIGHT_MAZE=[[1,1,1,1,1],[1,0,0,0,1],[1,1,1,1,1]]
    gym.register_envs(gymnasium_robotics)
    env_id="PointMaze_UMaze-v3"
    base_env=gym.make(env_id,maze_map=STRAIGHT_MAZE,max_episode_steps=100)
    env=RoboGymObservationWrapper(base_env)
    corners=detect_corners(env,50)
    obs,_=env.reset()
    state_dim=obs['observation'].shape[0]; goal_dim=obs['desired_goal'].shape[0]
    act_space=env.action_space
    agent=Agent(num_inputs=state_dim,action_space=act_space,goal_dim=goal_dim,
                gamma=0.98,tau=0.005,alpha=0.2,target_update_interval=1,
                hidden_size=256,learning_rate=3e-4,exploration_scaling_factor=1.0,
                her_strategy='future',her_k=4,use_langevin=True,corners=corners)
    buffer=ReplayBuffer(max_size=500_000,input_size=state_dim,
                        n_actions=act_space.shape[0],goal_size=goal_dim)
    tracker=CSVTracker("wvf_all_corners_mismatch")
    print("\nStarting WVF Training (internal goals + WVF rewards + HER)\n")
    train(agent,env,buffer,tracker,corners,tb_run_name="runs/wvf_all_corners")
# sac_baseline_heatmap.py
# ---------------------------------------------------------------------
# Evaluate trained SAC+HER agent over a 2D grid of target goals
# ---------------------------------------------------------------------
import numpy as np
import torch
import gymnasium as gym
import gymnasium_robotics
import matplotlib.pyplot as plt

from agent import Agent  # from your baseline implementation


# ---------------------------------------------------------------------
# Run one deterministic rollout to a given goal
# ---------------------------------------------------------------------
def rollout_to_goal(agent, env, goal, max_steps=60):
    obs, _ = env.reset()
    state = obs["observation"]
    for _ in range(max_steps):
        action = agent.select_action(state, goal, eval_mode=True)
        next_obs, _, done, truncated, _ = env.step(action)
        state = next_obs["observation"]
        achieved_goal = next_obs["achieved_goal"]

        # Check if goal achieved
        if np.linalg.norm(achieved_goal - goal) <= agent.success_threshold:
            return True

        if done or truncated:
            break
    return False


# ---------------------------------------------------------------------
# Evaluate over a 2D grid of goals and produce success map
# ---------------------------------------------------------------------
def evaluate_goal_grid(agent, env, y_range=(-0.75, 0.75), x_range=(-1.75, 1.75), n_points=51):
    xs = np.linspace(x_range[0], x_range[1], n_points)
    ys = np.linspace(y_range[0], y_range[1], n_points)
    success_map = np.zeros((n_points, n_points), dtype=np.float32)

    print(f"[eval] Scanning {n_points}×{n_points} goals within X∈{x_range}, Y∈{y_range}...")
    for i, gx in enumerate(xs):
        for j, gy in enumerate(ys):
            goal = np.array([gx, gy], dtype=np.float32)
            success = rollout_to_goal(agent, env, goal)
            success_map[j, i] = success  # transpose for correct plotting
            print(f"  goal=({gx:.2f},{gy:.2f}) → success={int(success)}")
    return xs, ys, success_map


# ---------------------------------------------------------------------
# Plot 2D heatmap of success over goal space
# ---------------------------------------------------------------------
def plot_success_heatmap(xs, ys, success_map):
    plt.figure(figsize=(6, 5))
    plt.imshow(
        np.flipud(success_map),
        extent=[xs[0], xs[-1], ys[0], ys[-1]],
        cmap="viridis",
        vmin=0,
        vmax=1,
        interpolation="nearest",
        aspect="equal",
    )
    plt.colorbar(label="Success (1 = reached goal)")
    plt.title("SAC+HER Goal Success Heatmap")
    plt.xlabel("Goal X")
    plt.ylabel("Goal Y")
    plt.grid(False)
    plt.show()


# ---------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------
if __name__ == "__main__":
    STRAIGHT_MAZE = [
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
    ]

    gym.register_envs(gymnasium_robotics)
    env_id = "PointMaze_UMaze-v3"
    base_env = gym.make(env_id, max_episode_steps=100, maze_map=STRAIGHT_MAZE)

    # Reset env and extract dimensions
    obs, _ = base_env.reset()
    state_dim = obs["observation"].shape[0]
    goal_dim = obs["desired_goal"].shape[0]
    act_space = base_env.action_space

    # Initialize agent (same structure as during training)
    agent = Agent(
        state_dim=state_dim,
        action_space=act_space,
        goal_dim=goal_dim,
        gamma=0.98,
        tau=0.005,
        lr=3e-4,
        alpha=0.2,
        her_k=4,
        success_threshold=0.5,
    )

    print("\nLoading trained SAC+HER checkpoint...")
    agent.load_checkpoint("checkpoints")

    # Evaluate over restricted goal range
    xs, ys, success_map = evaluate_goal_grid(
        agent, base_env, y_range=(-0.75, 0.75), x_range=(-1.75, 1.75), n_points=51
    )

    # Compute statistics
    mean_success = np.mean(success_map)
    std_success = np.std(success_map)

    print("\n--- Summary Statistics ---")
    print(f"Mean success rate: {mean_success:.3f}")
    print(f"Standard deviation: {std_success:.3f}")

    # Plot heatmap
    plot_success_heatmap(xs, ys, success_map)

# wvf_inference_heatmap.py
import numpy as np
import torch
import gymnasium as gym
import gymnasium_robotics
import matplotlib.pyplot as plt

from gym_robotics_custom import RoboGymObservationWrapper
from agent import Agent


# ---------------------------------------------------------------------
# Run one deterministic rollout to a specific goal
# ---------------------------------------------------------------------
def rollout_to_goal(agent, env, goal, max_steps=60):
    obs, _ = env.reset()
    state = obs["observation"]
    for _ in range(max_steps):
        action = agent.select_action(state, goal, evaluate=True)
        next_obs, _, done, truncated, info = env.step(action)
        state = next_obs["observation"]
        ag = next_obs["achieved_goal"]
        if agent.is_goal_achieved(ag, goal):
            return True
        if done or truncated:
            break
    return False


# ---------------------------------------------------------------------
# Evaluate over a 2D grid of goals and produce success map
# ---------------------------------------------------------------------
def evaluate_goal_grid(agent, env, x_range=(-0.75, 0.75), y_range=(-1.75, 1.75), n_points=21):
    xs = np.linspace(x_range[0], x_range[1], n_points)
    ys = np.linspace(y_range[0], y_range[1], n_points)
    success_map = np.zeros((n_points, n_points), dtype=np.float32)

    print(f"[eval] Scanning {n_points}×{n_points} goals within X∈{x_range}, Y∈{y_range}...")
    for i, gx in enumerate(xs):
        for j, gy in enumerate(ys):
            goal = np.array([gx, gy], dtype=np.float32)
            success = rollout_to_goal(agent, env, goal)
            success_map[j, i] = success  # note transpose for plotting
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
    plt.title("Zero-Shot WVF Goal Inference Heatmap")
    plt.xlabel("Goal X")
    plt.ylabel("Goal Y")
    plt.grid(False)
    plt.show()


# ---------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------
if __name__ == "__main__":
    U_MAZE = [
    [1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1],
    ]


    gym.register_envs(gymnasium_robotics)
    env_id = "PointMaze_UMaze-v3"
    base_env = gym.make(env_id, max_episode_steps=100, maze_map=U_MAZE)
    env = RoboGymObservationWrapper(base_env)

    obs, _ = env.reset()
    state_dim = obs["observation"].shape[0]
    goal_dim = obs["desired_goal"].shape[0]
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
    )

    print("\nLoading trained checkpoints...")
    agent.load_checkpoint("checkpoints")

    # Evaluate over a restricted grid of goals
    xs, ys, success_map = evaluate_goal_grid(
        agent, env, y_range=(-0.75, 0.75), x_range=(-1.75, 1.75), n_points=51
    )

    # -----------------------------------------------------------------
    # Compute mean and standard deviation
    # -----------------------------------------------------------------
    mean_success = np.mean(success_map)
    std_success = np.std(success_map)

    print("\n--- Summary Statistics ---")
    print(f"Mean success rate: {mean_success:.3f}")
    print(f"Standard deviation: {std_success:.3f}")

    # -----------------------------------------------------------------
    # Show results
    # -----------------------------------------------------------------
    plot_success_heatmap(xs, ys, success_map)

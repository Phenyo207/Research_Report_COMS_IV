# evaluate_fetchreach_mean_std.py

import numpy as np
import gymnasium as gym
import gymnasium_robotics

from gym_robotics_custom import RoboGymObservationWrapper
from main import CustomFetchReachRewardWrapper
from agent import Agent


# -------------------------------------------------------------
# One deterministic rollout to the given goal
# -------------------------------------------------------------
def rollout_to_goal(agent, env, goal, max_steps=50):
    obs, _ = env.reset()
    
    # Overwrite the environment's randomly sampled goal
    env.unwrapped.goal = goal.copy()
    obs["desired_goal"] = goal.copy()

    state = obs["observation"]

    for _ in range(max_steps):
        action = agent.select_action(state, goal, evaluate=True)
        next_obs, _, done, truncated, info = env.step(action)

        state = next_obs["observation"]
        achieved = next_obs["achieved_goal"]

        if agent.is_goal_achieved(achieved, goal):
            return True

        if done or truncated:
            break

    return False


# -------------------------------------------------------------
# Evaluation using real FetchReach goal sampling
# -------------------------------------------------------------
def evaluate_fetchreach(agent, env, n_goals=200):
    successes = []

    for _ in range(n_goals):
        # Let FetchReach sample a natural goal
        obs, _ = env.reset()
        goal = obs["desired_goal"].copy()

        success = rollout_to_goal(agent, env, goal)
        successes.append(success)

        print(f"Goal {goal} → {int(success)}")

    return np.array(successes)


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------
if __name__ == "__main__":
    gym.register_envs(gymnasium_robotics)

    # Environment
    base_env = gym.make("FetchReach-v4")
    base_env = CustomFetchReachRewardWrapper(base_env)
    env = RoboGymObservationWrapper(base_env)

    # Extract dims
    obs, _ = env.reset()
    num_inputs = obs["observation"].shape[0]
    goal_dim = obs["desired_goal"].shape[0]
    action_space = env.action_space

    # Agent (matches your agent.py EXACTLY)
    agent = Agent(
        num_inputs=num_inputs,
        action_space=action_space,
        goal_dim=goal_dim,
        gamma=0.98,
        tau=0.005,
        alpha=0.2,
        target_update_interval=1,
        hidden_size=256,
        learning_rate=3e-4,
        exploration_scaling_factor=1.0,
    )

    print("\nLoading checkpoint...")
    agent.load_checkpoint("checkpoints")

    # Evaluate
    results = evaluate_fetchreach(agent, env, n_goals=200)

    mean_success = np.mean(results)
    std_success = np.std(results)

    print("\n--- FetchReach Performance ---")
    print(f"Mean success: {mean_success:.3f}")
    print(f"Std dev:      {std_success:.3f}")

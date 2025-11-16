import numpy as np
import gymnasium as gym
import gymnasium_robotics

from gym_robotics_custom import RoboGymObservationWrapper
from agent import Agent


# ------------------------------------------------------------
# Roll out one attempt to reach a given 3D goal
# ------------------------------------------------------------
def rollout_to_goal(agent, env, goal, max_steps=50):
    obs, _ = env.reset()

    # Overwrite the sampled desired goal
    obs["desired_goal"] = goal.copy()
    env.unwrapped.goal = goal.copy()   # lets the env compute correct reward

    state = obs["observation"]

    for _ in range(max_steps):
        action = agent.select_action(state, goal, evaluate=True)
        next_obs, _, done, truncated, info = env.step(action)

        state = next_obs["observation"]
        achieved = next_obs["achieved_goal"]

        # success based on your agent's compute_reward
        if agent.compute_reward(achieved, goal) > 0:
            return True

        if done or truncated:
            break

    return False


# ------------------------------------------------------------
# Evaluate N goals sampled from env.reset()
# ------------------------------------------------------------
def evaluate_random_goals(agent, env, n_goals=200):
    success_list = []

    for i in range(n_goals):
        obs, _ = env.reset()
        goal = obs["desired_goal"].copy()     # ← THIS is the correct goal source

        success = rollout_to_goal(agent, env, goal)
        success_list.append(success)

        print(f"Goal {i+1}/{n_goals}  {goal} → {int(success)}")

    return np.array(success_list)


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
if __name__ == "__main__":

    gym.register_envs(gymnasium_robotics)

    env = gym.make("FetchReach-v4")
    env = RoboGymObservationWrapper(env)

    obs, _ = env.reset()

    # dims
    state_dim = obs["observation"].shape[0]
    goal_dim  = obs["desired_goal"].shape[0]
    action_dim = env.action_space.shape[0]

    # --------------------------------------------------------
    # Construct the agent using YOUR ACTUAL signature
    # --------------------------------------------------------
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
        device="cpu"
    )

    print("\nLoading trained checkpoint...")
    agent.load_checkpoint()

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------
    successes = evaluate_random_goals(agent, env, n_goals=250)

    mean_success = np.mean(successes)
    std_success  = np.std(successes)

    print("\n--- FINAL RESULTS ---")
    print(f"Mean success rate: {mean_success:.3f}")
    print(f"Std deviation:     {std_success:.3f}")

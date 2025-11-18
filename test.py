# test.py
"""
Testing script for WVF Agent (codeset2) - Straight Maze Onlyx
"""

import gymnasium as gym
import gymnasium_robotics
import numpy as np
import os
from gym_robotics_custom import RoboGymObservationWrapper
from agent import Agent

def test_agent(agent, env, episodes=10, max_episode_steps=100, verbose=True):
    """Test the WVF agent and collect statistics"""
    print(f"\n{'='*60}")
    print(f"Testing agent for {episodes} episodes...")
    print(f"{'='*60}\n")
    
    success_count = 0
    total_rewards = []
    episode_lengths = []
    goal_distances = []
    done_actions_used = 0
    
    for ep in range(episodes):
        obs, info = env.reset()
        state = obs['observation']
        desired_goal = obs['desired_goal']
        achieved_goal = obs['achieved_goal']
        
        # Use goal policy for testing
        if len(agent.internal_goals) > 0:
            chosen_goal, goal_type = agent.choose_goal_for_episode(desired_goal, state)
        else:
            chosen_goal = desired_goal
            goal_type = 'environment'
        
        episode_reward = 0.0
        episode_steps = 0
        episode_success = False
        episode_done_actions = 0
        
        if verbose:
            print(f"Episode {ep+1}/{episodes}")
            print(f"  Goal type: {goal_type}")
            print(f"  Target: {chosen_goal}")
        
        while episode_steps < max_episode_steps:
            action = agent.select_action(state, chosen_goal, evaluate=True)
            next_obs, _, done, truncated, info = env.step(action)
            next_state = next_obs['observation']
            next_ag = next_obs['achieved_goal']
            
            if info.get('agent_chose_done', False):
                episode_done_actions += 1
                done_actions_used += 1
            
            wvf_reward = agent.compute_reward(next_ag, chosen_goal, info)
            
            if agent.is_goal_achieved(next_ag, chosen_goal):
                episode_success = True
                if verbose:
                    print(f"  SUCCESS at step {episode_steps+1}!")
                break
            
            episode_reward += wvf_reward
            episode_steps += 1
            state = next_state
            achieved_goal = next_ag
            
            if done or truncated:
                break
        
        final_distance = np.linalg.norm(achieved_goal - chosen_goal)
        
        if episode_success:
            success_count += 1
        total_rewards.append(episode_reward)
        episode_lengths.append(episode_steps)
        goal_distances.append(final_distance)
        
        if verbose:
            print(f"  Steps: {episode_steps}, Reward: {episode_reward:.2f}, "
                  f"Distance: {final_distance:.3f}, Done actions: {episode_done_actions}")
            print()
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Success Rate: {success_count}/{episodes} = {100*success_count/episodes:.1f}%")
    print(f"Average Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Average Episode Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Average Final Distance: {np.mean(goal_distances):.3f} ± {np.std(goal_distances):.3f}")
    print(f"Done Actions Used: {done_actions_used} ({100*done_actions_used/sum(episode_lengths):.1f}% of actions)")
    print(f"\nAgent Statistics:")
    print(f"  Internal Goals Discovered: {len(agent.internal_goals)}")
    print(f"  Successful Goals: {len(agent.successful_goals)}")
    print(f"  Lifetime Done Action Ratio: {agent.get_done_action_ratio():.3f}")
    
    mastery_stats = agent.get_mastery_stats()
    if mastery_stats['current_mastery'] > 0:
        print(f"\nMastery Statistics:")
        print(f"  Current: {mastery_stats['current_mastery']:.3f}")
        print(f"  Best: {mastery_stats['best_mastery']:.3f}")
        print(f"  Average: {mastery_stats['avg_mastery']:.3f}")
    
    if len(agent.internal_goals) > 0:
        print(f"\nSample Internal Goals (first 10):")
        for i, goal in enumerate(list(agent.internal_goals)[:10]):
            print(f"  {i+1}. {goal}")
    
    print(f"{'='*60}\n")
    
    return {
        'success_rate': success_count / episodes,
        'avg_reward': np.mean(total_rewards),
        'avg_length': np.mean(episode_lengths),
        'avg_distance': np.mean(goal_distances)
    }


if __name__ == '__main__':
    # Configuration
    env_name = "PointMaze_UMaze-v3"
    max_episode_steps = 100
    test_episodes = 20
    hidden_size = 256
    learning_rate = 3e-4
    gamma = 0.98
    tau = 0.005
    alpha = 0.2
    checkpoint_dir = 'checkpoints'
    
    # Straight maze (same as training)
    STRAIGHT_MAZE = [
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
    ]
    
    print(f"\n{'='*60}")
    print("WVF Agent Testing - Straight Maze")
    print(f"{'='*60}")
    print(f"Environment: {env_name}")
    print(f"Max Episode Steps: {max_episode_steps}")
    print(f"Test Episodes: {test_episodes}")
    print(f"{'='*60}\n")
    
    # Check checkpoints
    required_files = ['critic.pt', 'critic_target.pt', 'policy.pt']
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(checkpoint_dir, f))]
    
    if missing_files:
        print(f"ERROR: Missing checkpoint files: {missing_files}")
        print(f"Please train the agent first using main.py")
        exit(1)
    
    # Create environment
    gym.register_envs(gymnasium_robotics)
    base_env = gym.make(env_name, max_episode_steps=max_episode_steps, 
                       render_mode='human', maze_map=STRAIGHT_MAZE)
    env = RoboGymObservationWrapper(base_env)
    
    obs, _ = env.reset()
    state_dim = obs['observation'].shape[0]
    goal_dim = obs['desired_goal'].shape[0]
    
    print(f"Dimensions: State={state_dim}, Goal={goal_dim}, Action={env.action_space.shape[0]}\n")
    
    # Create and load agent
    agent = Agent(
        num_inputs=state_dim,
        action_space=env.action_space,
        goal_dim=goal_dim,
        gamma=gamma,
        tau=tau,
        alpha=alpha,
        target_update_interval=1,
        hidden_size=hidden_size,
        learning_rate=learning_rate,
        exploration_scaling_factor=1.0,
        her_strategy='future',
        her_k=4
    )
    
    print("Loading checkpoint...")
    try:
        agent.load_checkpoint(checkpoint_dir)
        print(f"Loaded successfully!")
        print(f"Reward structure: Success={agent.goal_achievement_reward}, "
              f"Wrong-Done={agent.R_MIN}, Step={agent.step_reward}\n")
    except Exception as e:
        print(f"ERROR: {e}")
        env.close()
        exit(1)
    
    # Test
    results = test_agent(agent, env, episodes=test_episodes, 
                        max_episode_steps=max_episode_steps, verbose=True)
    
    env.close()
    print("Testing complete!\n")
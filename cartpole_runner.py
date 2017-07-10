import gym
import ipdb
import numpy as np

gym.envs.register(
    id='CartPole-v000',
    entry_point='gym.envs.classic_control:CartPoleEnv',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 300},
    reward_threshold=1000.0,
)

def run_n_cartpole_simulations(n_episodes, action_sampler, env=None, render=False):
    if env is None:
        env = gym.make('CartPole-v1')
    if render:
        env.render()
    encoded_actions = np.array([[1,0],[0,1]])
    experience_replay_buffer = list()
    episode_lengths = list()
    episode_rewards = list()
    for c_episode in range(0, n_episodes):
        state = env.reset()
        done = False
        episode_length = 0
        episode_reward = 0
        actions = list()
        # Run one episode
        while not done:
            if render:
                env.render()
            action = action_sampler(env.action_space, np.array(state, ndmin=2))
            actions.append(action)
            next_state, reward, done, info = env.step(action)
            reward = -1 if done else 0
            experience_replay_buffer.append((state, encoded_actions[action,:], reward, next_state))
            episode_length += 1
            episode_reward += reward
            state = next_state
        episode_lengths.append(episode_length)
        episode_rewards.append(episode_reward)
    return experience_replay_buffer, episode_lengths, episode_rewards

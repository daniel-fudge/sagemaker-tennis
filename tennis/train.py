"""
This script trains and saves the model and plots its performance.

Note:  You need to verify the env path is correct for you PC and OS.
"""

from collections import deque
from tennis.ddpg_agent import Agent
import numpy as np
import torch


def make_plot(show=False):
    """Makes a pretty training plot call score.png.

    Args:
        show (bool):  If True, show the image.  If False, save the image.
    """

    import matplotlib.pyplot as plt

    target = 0.5

    # Load the previous scores and calculated running mean of 100 runs
    # ---------------------------------------------------------------------------------------
    with np.load('scores.npz') as data:
        scores = data['arr_0']
    cum_sum = np.cumsum(np.insert(scores, 0, 0))
    rolling_mean = (cum_sum[100:] - cum_sum[:-100]) / 100

    # Make a pretty plot
    # ---------------------------------------------------------------------------------------
    plt.figure()
    x_max = len(scores)
    y_min = scores.min() - 1
    x = np.arange(x_max)
    plt.scatter(x, scores, s=2, c='k', label='Raw Scores', zorder=4)
    plt.plot(x[99:], rolling_mean, lw=2, label='Rolling Mean', zorder=3)
    plt.scatter(x_max, rolling_mean[-1], c='g', s=40, marker='*', label='Episode {}'.format(x_max), zorder=5)
    plt.plot([0, x_max], [target, target], lw=1, c='grey', ls='--', label='Target Score = {}'.format(target), zorder=1)
    plt.plot([x_max, x_max], [y_min, rolling_mean[-1]], lw=1, c='grey', ls='--', label=None, zorder=2)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.legend()
    plt.xlim([0, x_max + 5])
    plt.ylim(bottom=0)
    if show:
        plt.show()
    else:
        plt.savefig('scores.png', dpi=200)
    plt.close()


def train(agents, env, n_episodes=2000, max_t=1000):
    """This function trains the given agent in the given environment.

    Args:
        agents (list of Agent):  The 2 agents to train.
        env (unityagents.UnityEnvironment):  The training environment
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of time steps per episode
    """

    scores = list()
    scores_window = deque(maxlen=100)
    brain_name = env.brain_names[0]
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        score = 0
        for t in range(max_t):
            actions = [agents[i].act(state=states[i]) for i in range(2)]
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            done_values = env_info.local_done
            for i in range(2):
                agents[i].step(states, actions, rewards, next_states, done_values)
            states = next_states
            score += max(rewards)
            if np.any(done_values):
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= 0.5:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode,
                                                                                         np.mean(scores_window)))
            break

    # Save models weights and scores
    for i in range(2):
        torch.save(agents[i].actor_target.state_dict(), 'checkpoint_actor_{}.pth'.format(i + 1))
        torch.save(agents[i].critic_target.state_dict(), 'checkpoint_critic_{}.pth'.format(i + 1))
    np.savez('scores.npz', scores)


def setup(env):
    """Setups up the environment to train.

    Args:
        env (unityagents.UnityEnvironment):  The training environment

    Returns:
        list of Agent:  Agents for player #1 and #2
    """
    # Setup the environment and print of some information for reference
    # -----------------------------------------------------------------------------------
    print('Setting up the environment.')
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)

    # examine the state space
    state_size = env_info.vector_observations.shape[1]
    print('State space per agent: {}'.format(state_size))

    # Setup the agent and return it
    # -----------------------------------------------------------------------------------
    print('Setting up agent #1 and #2.')
    return [Agent(state_size=state_size, action_size=action_size, random_seed=42) for _ in range(2)]

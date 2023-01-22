import numpy as np
import time

from maci.misc import logger
from copy import deepcopy


def rollout(env, policy, path_length, render=False, speedup=None):
    Da = env.action_space.flat_dim
    Do = env.observation_space.flat_dim

    observation = env.reset()
    policy.reset()

    observations = np.zeros((path_length + 1, Do))
    actions = np.zeros((path_length, Da))
    terminals = np.zeros((path_length, ))
    rewards = np.zeros((path_length, ))
    agent_infos = []
    env_infos = []

    t = 0
    for t in range(path_length):

        action, agent_info = policy.get_action(observation)
        next_obs, reward, terminal, env_info = env.step(action)

        agent_infos.append(agent_info)
        env_infos.append(env_info)

        actions[t] = action
        terminals[t] = terminal
        rewards[t] = reward
        observations[t] = observation

        observation = next_obs

        if render:
            env.render()
            time_step = 0.05
            time.sleep(time_step / speedup)

        if terminal:
            break

    observations[t + 1] = observation

    path = {
        'observations': observations[:t + 1],
        'actions': actions[:t + 1],
        'rewards': rewards[:t + 1],
        'terminals': terminals[:t + 1],
        'next_observations': observations[1:t + 2],
        'agent_infos': agent_infos,
        'env_infos': env_infos
    }

    return path


def rollouts(env, policy, path_length, n_paths):
    paths = [
        rollout(env, policy, path_length)
        for i in range(n_paths)
    ]

    return paths


class Sampler(object):
    def __init__(self, max_path_length, min_pool_size, batch_size):
        self._max_path_length = max_path_length
        self._min_pool_size = min_pool_size
        self._batch_size = batch_size

        self.env = None
        self.policy = None
        self.pool = None

    def initialize(self, env, policy, pool):
        self.env = env
        self.policy = policy
        self.pool = pool

    def set_policy(self, policy):
        self.policy = policy

    def sample(self):
        raise NotImplementedError

    def batch_ready(self):
        enough_samples = self.pool.size >= self._min_pool_size
        return enough_samples

    def random_batch(self):
        return self.pool.random_batch(self._batch_size)

    def terminate(self):
        self.env.terminate()

    def log_diagnostics(self):
        logger.record_tabular('pool-size', self.pool.size)


class SimpleSampler(Sampler):
    def __init__(self, **kwargs):
        super(SimpleSampler, self).__init__(**kwargs)

        self._path_length = 0
        self._path_return = 0
        self._last_path_return = 0
        self._max_path_return = -np.inf
        self._n_episodes = 0
        self._current_observation = None
        self._total_samples = 0

    def sample(self):
        if self._current_observation is None:
            self._current_observation = self.env.reset()

        action, _ = self.policy.get_action(self._current_observation)
        next_observation, reward, terminal, info = self.env.step(action)
        self._path_length += 1
        self._path_return += reward
        self._total_samples += 1

        self.pool.add_sample(
            observation=self._current_observation,
            action=action,
            reward=reward,
            terminal=terminal,
            next_observation=next_observation)

        if terminal or self._path_length >= self._max_path_length:
            self.policy.reset()
            self._current_observation = self.env.reset()
            self._path_length = 0
            self._max_path_return = max(self._max_path_return,
                                        self._path_return)
            self._last_path_return = self._path_return

            self._path_return = 0
            self._n_episodes += 1

        else:
            self._current_observation = next_observation

    def log_diagnostics(self):
        super(SimpleSampler, self).log_diagnostics()
        logger.record_tabular('max-path-return', self._max_path_return)
        logger.record_tabular('last-path-return', self._last_path_return)
        logger.record_tabular('episodes', self._n_episodes)
        logger.record_tabular('total-samples', self._total_samples)


class MASampler(SimpleSampler):
    """
The MASampler class is a subclass of SimpleSampler, which is a class for collecting samples from an environment using a
given set of policies. MASampler stands for "Multi-Agent Sampler", and it is used for sampling from an environment with
multiple agents.

In the __init__ method, the superclass SimpleSampler is initialized using the **kwargs syntax, which allows the method
to accept any additional keyword arguments that might be defined in the superclass. The __init__ method of MASampler then
initializes some additional attributes:

self.agent_num: the number of agents in the environment
self.joint: a boolean indicating whether the replay buffer should store information about the actions of all agents (True)
or only the actions of a single agent (False)
self._path_length: the length of the current path, i.e. the number of steps taken in the environment since the last reset
self._path_return: an array storing the accumulated rewards for each agent at the current path
self._last_path_return: an array storing the accumulated rewards for each agent at the previous path
self._max_path_return: an array storing the maximum accumulated rewards for each agent across all paths
self._n_episodes: the number of episodes (i.e. environment resets) that have occurred during the sampling process
self._total_samples: the total number of samples collected during the sampling process
self._current_observation_n: the current observations for each agent
self.env: the environment used for sampling
self.agents: the agents used for sampling
The set_policy method allows you to set the policy for each of the agents. It takes a list of policies as input,
and assigns each policy to the corresponding agent.

The batch_ready method returns True if there are at least self._min_pool_size samples in the replay buffer of each
of the agents, and False otherwise. The self._min_pool_size attribute is defined in the superclass SimpleSampler and
determines the minimum number of samples that need to be present in the replay buffer before the sampling process can start.

The random_batch method returns a random batch of self._batch_size samples from the replay buffer of the agent with the
given index. The self._batch_size attribute is also defined in the superclass SimpleSampler and determines the number of
samples to be included in each batch.

The initialize method is used to set the environment and the agents that will be used in the sampling process. It takes
an env object and a list of agents as input, and assigns them to the corresponding attributes of the MASampler instance.

The sample method is used to collect samples from the environment. It first determines the actions to be taken by each
of the agents based on their current policies and observations. It does this by calling the get_action method of the policy object for each agent, passing the current observation as input. If the agent has a joint policy (i.e. a policy that takes into account the actions of all other agents), it takes only the first agent._action_dim elements of the action
    """
    def __init__(self, agent_num, joint, **kwargs):
        super(SimpleSampler, self).__init__(**kwargs)
        self.agent_num = agent_num
        self.joint = joint
        self._path_length = 0
        self._path_return = np.array([0.] * self.agent_num, dtype=np.float32)
        self._last_path_return = np.array([0.] * self.agent_num, dtype=np.float32)
        self._max_path_return = np.array([-np.inf] * self.agent_num, dtype=np.float32)
        self._n_episodes = 0
        self._total_samples = 0

        self._current_observation_n = None
        self.env = None
        self.agents = None

    def set_policy(self, policies):
        tahas_agents = [agent for cluster in self.agents.values() for agent in cluster]
        for agent, policy in zip(tahas_agents, policies):
            agent.policy = policy

    def batch_ready(self):
        tahas_agents = [agent for cluster in self.agents.values() for agent in cluster]
        enough_samples = tahas_agents[0].pool.size >= self._min_pool_size
        return enough_samples

    def random_batch(self, i):
        tahas_agents = [agent for cluster in self.agents.values() for agent in cluster]
        return tahas_agents[i].pool.random_batch(self._batch_size)

    def initialize(self, env, agents):
        self._current_observation_n = None
        self.env = env
        self.agents = agents

    def sample(self, clusters: dict = None):
        if clusters:
            pass
        if self._current_observation_n is None:
            self._current_observation_n = self.env.reset()
        action_n = []
        tahas_agents = [agent for cluster in self.agents.values() for agent in cluster]
        for agent, current_observation in zip(tahas_agents, self._current_observation_n):
            action, _ = agent.policy.get_action(current_observation)
            if agent.joint_policy:
                action_n.append(np.array(action)[0:agent._action_dim])
            else:
                action_n.append(np.array(action))
        next_observation_n, reward_n, done_n, info = self.env.step(action_n)
        self._path_length += 1
        self._path_return += np.array(reward_n, dtype=np.float32)
        self._total_samples += 1

        i = -1
        for cluster, tahas_agents in enumerate(self.agents.values()):
            for agent in tahas_agents:
                i += 1
                action = deepcopy(action_n[i])
                if agent.pool.joint:
                    opponent_action = deepcopy(action_n)
                    for _ in clusters[cluster]:
                        del opponent_action[i]
                    opponent_action = np.array(opponent_action).flatten()
                    agent.pool.add_sample(observation=self._current_observation_n[i],
                                          action=action,
                                          reward=reward_n[i],
                                          terminal=done_n[i],
                                          next_observation=next_observation_n[i],
                                          opponent_action=opponent_action)
                else:
                    agent.pool.add_sample(observation=self._current_observation_n[i],
                                          action=action,
                                          reward=reward_n[i],
                                          terminal=done_n[i],
                                          next_observation=next_observation_n[i])

        if np.all(done_n) or self._path_length >= self._max_path_length:
            self._current_observation_n = self.env.reset()
            self._max_path_return = np.maximum(self._max_path_return, self._path_return)
            self._mean_path_return = self._path_return / self._path_length
            self._last_path_return = self._path_return

            self._path_length = 0

            self._path_return = np.array([0.] * self.agent_num, dtype=np.float32)
            self._n_episodes += 1

            self.log_diagnostics()
            logger.dump_tabular(with_prefix=False)

        else:
            self._current_observation_n = next_observation_n

    def log_diagnostics(self):
        for i in range(self.agent_num):
            logger.record_tabular('max-path-return_agent_{}'.format(i), self._max_path_return[i])
            logger.record_tabular('mean-path-return_agent_{}'.format(i), self._mean_path_return[i])
            logger.record_tabular('last-path-return_agent_{}'.format(i), self._last_path_return[i])
        logger.record_tabular('episodes', self._n_episodes)
        logger.record_tabular('total-samples', self._total_samples)



class DummySampler(Sampler):
    def __init__(self, batch_size, max_path_length):
        super(DummySampler, self).__init__(
            max_path_length=max_path_length,
            min_pool_size=0,
            batch_size=batch_size)

    def sample(self):
        pass

import numpy as np
import tensorflow as tf
from maci.learners import MADDPG, MAVBAC, MASQL
from maci.misc.kernel import adaptive_isotropic_gaussian_kernel
from maci.replay_buffers import SimpleReplayBuffer
from maci.value_functions.sq_value_function import NNQFunction, NNJointQFunction
from maci.policies import StochasticNNConditionalPolicy, StochasticNNPolicy
from maci.policies.deterministic_policy import DeterministicNNPolicy, ConditionalDeterministicNNPolicy, DeterministicToMNNPolicy
from maci.policies.uniform_policy import UniformPolicy
from maci.policies.level_k_policy import MultiLevelPolicy, GeneralizedMultiLevelPolicy


def masql_agent(model_name, i, env, M, u_range, base_kwargs, game_name='matrix', clusters_schema=None):
    joint = True
    squash = True
    squash_func = tf.tanh
    sampling = False
    if 'particle' in game_name:
        sampling = True
        squash_func = tf.nn.softmax

    pool = SimpleReplayBuffer(env.env_specs, max_replay_buffer_size=1e4, joint=joint, agent_id=i, _clusters=clusters_schema)
    policy = StochasticNNPolicy(env.env_specs,
                                hidden_layer_sizes=(M, M),
                                squash=squash, squash_func=squash_func, sampling=sampling, u_range=u_range, joint=joint,
                                agent_id=i, _clusters=clusters_schema)

    qf = NNQFunction(env_spec=env.env_specs, hidden_layer_sizes=[M, M], joint=joint, agent_id=i, _clusters=clusters_schema, not_target = True)
    target_qf = NNQFunction(env_spec=env.env_specs, hidden_layer_sizes=[M, M], name='target_qf', joint=joint,
                            agent_id=i, _clusters=clusters_schema)

    plotter = None

    agent = MASQL(
        base_kwargs=base_kwargs,
        agent_id=i,
        env=env,
        pool=pool,
        qf=qf,
        target_qf=target_qf,
        policy=policy,
        plotter=plotter,
        policy_lr=3e-4,
        qf_lr=3e-4,
        tau=0.01,
        value_n_particles=16,
        td_target_update_interval=10,
        kernel_fn=adaptive_isotropic_gaussian_kernel,
        kernel_n_particles=32,
        kernel_update_ratio=0.5,
        discount=0.99,
        reward_scale=1,
        save_full_state=False,
        clusters_schema=clusters_schema)
    return agent


def get_level_k_policy(env, k, M, agent_id, u_range, opponent_conditional_policy, game_name='pbeauty'):
    """
    Taha:
    This function appears to define and return two MultiLevelPolicy objects: one called k_policy and the other called
    target_k_policy. These objects are used to represent policies for agents in a multi-agent environment.

    The MultiLevelPolicy class is a type of policy that models the decision-making process of an agent in a multi-agent
    environment as a hierarchy of policies. At each level of the hierarchy, the agent can either choose a simple base
    policy or a more complex conditional policy, which takes into account the actions and observations of other agents.
    The k parameter specifies the number of levels in the hierarchy.

    The function takes several arguments as input:

    env: an environment object, which specifies the characteristics of the environment in which the agents operate.
    k: an integer representing the number of levels in the hierarchy of policies for each agent.
    M: an integer representing the size of the hidden layers of the neural networks used to define the conditional policies.
    agent_id: an integer representing the unique identifier of the agent for whom the policies are being defined.
    u_range: a tuple specifying the range of valid actions for the agent.
    opponent_conditional_policy: a conditional policy object representing the policy of an opponent of the agent.
    game_name: a string representing the name of the game being played.
    The function first defines several variables based on the value of game_name. Then, it creates a base policy object
    called base_policy using the UniformPolicy class. This base policy represents a simple policy that uniformly selects
    actions from the specified range.

    The function then creates a conditional policy object called conditional_policy using the
    ConditionalDeterministicNNPolicy class. This policy represents a more complex policy that is learned through training
    and takes into account the actions and observations of other agents.

    Finally, the function creates two MultiLevelPolicy objects: k_policy and target_k_policy. These objects represent the
    hierarchy of policies for the agent, with k levels and the specified base and conditional policies.
    The target_k_policy object is created using the same parameters as k_policy, but with the tf.variable_scope set to
    'target_levelk_{}'.format(agent_id) to allow for separate variable scopes for the two policies.
    The function returns both k_policy and target_k_policy as output.
        """

    urange = [-1, 1.]
    if_softmax = False
    if 'particle' in game_name:
        urange = [-100., 100.]
        if_softmax = True
    squash = True
    squash_func = tf.tanh
    correct_tanh = True

    sampling = False
    if 'particle' in game_name:
        sampling = True
        squash_func = tf.nn.softmax
        correct_tanh = False
    # print('env spec', env.env_specs)
    opponent = False
    if k % 2 == 1:
        opponent = True

    base_policy = UniformPolicy(env.env_specs, agent_id=agent_id, opponent=opponent, urange=urange, if_softmax=if_softmax)
    conditional_policy = ConditionalDeterministicNNPolicy(env.env_specs,
                                                          hidden_layer_sizes=(M, M),
                                                          name='conditional_policy',
                                                          squash=squash, squash_func=squash_func, sampling=sampling, u_range=u_range, joint=False,
                                                          agent_id=agent_id)
    k_policy = MultiLevelPolicy(env_spec=env.env_specs,
                                k=k,
                                base_policy=base_policy,
                                conditional_policy=conditional_policy,
                                opponent_conditional_policy=opponent_conditional_policy,
                                agent_id=agent_id)
    with tf.variable_scope('target_levelk_{}'.format(agent_id), reuse=True):
        target_k_policy = MultiLevelPolicy(env_spec=env.env_specs,
                                    k=k,
                                    base_policy=base_policy,
                                    conditional_policy=conditional_policy,
                                    opponent_conditional_policy=opponent_conditional_policy,
                                    agent_id=agent_id)
    return k_policy, target_k_policy


def pr2ac_agent(model_name, i, env, M, u_range, base_kwargs, k=0, g=False, mu=1.5, game_name='matrix', aux=True):
    joint = False
    squash = True
    squash_func = tf.tanh
    correct_tanh = True
    sampling = False
    if 'particle' in game_name:
        sampling = True
        squash = True
        squash_func = tf.nn.softmax
        correct_tanh = False

    pool = SimpleReplayBuffer(env.env_specs, max_replay_buffer_size=1e4, joint=joint, agent_id=i)

    opponent_conditional_policy = StochasticNNConditionalPolicy(env.env_specs,
                                                       hidden_layer_sizes=(M, M),
                                                       name='opponent_conditional_policy',
                                                       squash=squash, squash_func=squash_func, sampling=sampling, u_range=u_range, joint=joint,
                                                       agent_id=i)


    if g:
        policies = []
        target_policies = []
        for kk in range(1, k+1):
            policy, target_policy = get_level_k_policy(env, kk, M, i, u_range, opponent_conditional_policy, game_name=game_name)
            policies.append(policy)
            target_policies.append(target_policy)
        policy = GeneralizedMultiLevelPolicy(env.env_specs, policies=policies, agent_id=i, k=k, mu=mu)
        target_policy = GeneralizedMultiLevelPolicy(env.env_specs, policies=policies, agent_id=i, k=k, mu=mu, correct_tanh=correct_tanh)
    else:
        if k == 0:
            policy = DeterministicNNPolicy(env.env_specs,
                                           hidden_layer_sizes=(M, M),
                                           squash=squash, squash_func=squash_func, sampling=sampling, u_range=u_range,
                                           joint=False, agent_id=i)
            target_policy = DeterministicNNPolicy(env.env_specs,
                                                  hidden_layer_sizes=(M, M),
                                                  name='target_policy',
                                                  squash=squash, squash_func=squash_func, sampling=sampling,
                                                  u_range=u_range, joint=False, agent_id=i)
        if k > 0:
            policy, target_policy = get_level_k_policy(env, k, M, i, u_range, opponent_conditional_policy, game_name=game_name)




    joint_qf = NNJointQFunction(env_spec=env.env_specs, hidden_layer_sizes=[M, M], joint=joint, agent_id=i)
    target_joint_qf = NNJointQFunction(env_spec=env.env_specs, hidden_layer_sizes=[M, M], name='target_joint_qf',
                                       joint=joint, agent_id=i)

    qf = NNQFunction(env_spec=env.env_specs, hidden_layer_sizes=[M, M], joint=False, agent_id=i)
    plotter = None

    agent = MAVBAC(
        base_kwargs=base_kwargs,
        agent_id=i,
        env=env,
        pool=pool,
        joint_qf=joint_qf,
        target_joint_qf=target_joint_qf,
        qf=qf,
        policy=policy,
        target_policy=target_policy,
        conditional_policy=opponent_conditional_policy,
        plotter=plotter,
        policy_lr=3e-4,
        qf_lr=3e-4,
        joint=False,
        value_n_particles=16,
        kernel_fn=adaptive_isotropic_gaussian_kernel,
        kernel_n_particles=32,
        kernel_update_ratio=0.5,
        td_target_update_interval=5,
        discount=0.99,
        reward_scale=1,
        tau=0.01,
        save_full_state=False,
        k=k,
        aux=aux)
    return agent


def ddpg_agent(joint, opponent_modelling, model_name, i, env, M, u_range, base_kwargs, game_name='matrix', clusters_schema=None):
    # joint = True
    # opponent_modelling = False
    print(model_name)
    squash = True
    squash_func = tf.tanh
    sampling = False

    if 'particle' in game_name:
        squash_func = tf.nn.softmax
        sampling = True

    print(joint, opponent_modelling)
    pool = SimpleReplayBuffer(env.env_specs, max_replay_buffer_size=1e4, joint=joint, agent_id=i, _clusters=clusters_schema)

    
        
    opponent_policy = None
    if opponent_modelling:
        opponent_policy = DeterministicNNPolicy(env.env_specs,
                                                hidden_layer_sizes=(M, M),
                                                name='opponent_policy',
                                                squash=squash, squash_func=squash_func, u_range=u_range, joint=True,
                                                opponent_policy=True,
                                                agent_id=i)
    if 'ToM' in model_name:
        policy = DeterministicToMNNPolicy(env.env_specs,
                                   hidden_layer_sizes=(M, M),
                                   cond_policy=opponent_policy,
                                   squash=squash, squash_func=squash_func, sampling=sampling,u_range=u_range, joint=False,
                                   agent_id=i)
        target_policy = DeterministicToMNNPolicy(env.env_specs,
                                          hidden_layer_sizes=(M, M),
                                          cond_policy=opponent_policy,
                                          name='target_policy',
                                          squash=squash, squash_func=squash_func,sampling=sampling, u_range=u_range,
                                          joint=False,
                                          agent_id=i)
    else:
        policy = DeterministicNNPolicy(env.env_specs,
                                   hidden_layer_sizes=(M, M),
                                   squash=squash, squash_func=squash_func, sampling=sampling,u_range=u_range, joint=False,
                                   agent_id=i)
        target_policy = DeterministicNNPolicy(env.env_specs,
                                          hidden_layer_sizes=(M, M),
                                          name='target_policy',
                                          squash=squash, squash_func=squash_func,sampling=sampling, u_range=u_range,
                                          joint=False,
                                          agent_id=i)
    qf = NNQFunction(env_spec=env.env_specs, hidden_layer_sizes=[M, M], joint=joint, agent_id=i, _clusters=clusters_schema, not_target = True)
    target_qf = NNQFunction(env_spec=env.env_specs, hidden_layer_sizes=[M, M], name='target_qf', joint=joint,
                            agent_id=i, _clusters=clusters_schema)
    plotter = None

    agent = MADDPG(
        base_kwargs=base_kwargs,
        agent_id=i,
        env=env,
        pool=pool,
        qf=qf,
        target_qf=target_qf,
        policy=policy,
        target_policy=target_policy,
        opponent_policy=opponent_policy,
        plotter=plotter,
        policy_lr=3e-4,
        qf_lr=3e-4,
        joint=joint,
        opponent_modelling=opponent_modelling,
        td_target_update_interval=10,
        discount=0.99,
        reward_scale=0.1,
        save_full_state=False,
        clusters_schema=clusters_schema)
    return agent
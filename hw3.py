import infrastructure.utils.torch_utils as tu
from infrastructure.utils.logger import Logger

import gymnasium as gym
import numpy as np

import torch
from torch.distributions import Categorical 
import torch.nn as nn
import torch.nn.functional as F 


"""
    Please fill these out with 
        a) The name of your agent for the leaderboard
        b) UCO's of the members of your team
"""

NAME = "AlgorithmName"
UCOS = [ 123456, 234567 ]

"""
    The familiar Policy/Trainer interface, including a new value method.
"""
class Policy:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError()

    # Should sample an action from the policy in the given state
    def play(self, state : int, *args, **kwargs):
        raise NotImplementedError()

    # Should return the predicted logits for the given state
    def raw(self, state: int, *args, **kwargs):
        raise NotImplementedError()

    # Should return the predicted value of the given state V(state)
    def value(self, state: int, *args, **kwargs):
        raise NotImplementedError()

class Trainer:
    def __init__(self, env, *args, **kwargs):
        self.env = env

    # `gamma` is the discount factor
    # `steps` is the total number of calls to env.step()
    def train(self, gamma : float, steps : int, *args, **kwargs) -> Policy:
        raise NotImplementedError()


"""
    You'll need to implement two models this time, one neural network for the
    value function (critic), and one for the policy (actor).

    In theory, it is possible to have a shared architecture for both networks,
    but that will most likely be more difficult to train.
"""
class ValueNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64):
        super(ValueNet, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)

        # Implement the network architecture, see torch.nn layers.
        
    def forward(self, x):
        # Add activation functions and such
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

    @torch.no_grad()
    def value_no_grad(self, obs):
        return self(obs)

    def value(self, obs):
        return self(obs)


class PolicyNet(nn.Module):

    # input ~ dimensions of state space, output ~ action count (discrete envs)
    def __init__(self, input_size, output_size, hidden_size=64):
        super(PolicyNet, self).__init__()
        self.dummy_layer = nn.Linear(1, 1)

    # `play` method assumes the forward returns logits
    def forward(self, x):
        return torch.zeros(2)

    @torch.no_grad()
    def play(self, obs):
        output = self(obs)
        dist = Categorical(logits=output)
        action = dist.sample()
        return action.item()
    
    def log_probs(self, obs, actions):
        output = self(obs)
        dist = Categorical(logits=output)
        return dist.log_prob(actions)
    
    @torch.no_grad()
    def log_probs_no_grad(self, obs, actions):
        self.log_probs(obs, actions)



"""
    The goal in this assignment is to implement a policy gradient agent.
    You'll start by implementing vanilla Policy gradient and incrementally
    adding features to the base algorithm.

    These features include:
        1) A neural network state value function critic.
        2) Generalized Advantage estimation, which utilizes the same value function.

    Afterwards, you are free to experiment with other improvements, for
    example:
        a) Importance sampling to reutilize old data
        b) Utilizing a PPO-style clipped loss
        c) Using TD(lambda) returns for learning the value function
        d) Normalized advantages
        e) Entropy regularization

    and many more, see resources in the assignment pdf.
"""


class PGPolicy(Policy):
    def __init__(self, net : PolicyNet, value_net : ValueNet):
        self.net = net
        self.value_net = value_net

    # Returns played action in state
    def play(self, state):
        return self.net.play(state)

    # Returns value
    def value(self, state):
        return self.value_net.value_no_grad(state)


def collect_trajectories(env, policy, step_limit, gamma, bootstrap_trunc):

    """
    This is a helper function that collects a batch of episodes,
    totalling `step_limit` in steps. The last episode is truncated to
    accomodate for the given limit.
    

    You can use this during training to get the necessary data for learning.

        Returns several flattened tensors:

            1) States encountered
            2) Actions played
            3) Rewards collected
            4) Dones - Points of termination / truncation.

        
        Whenever done[i] is True, then (states[i], actions[i], rewards[i]) is
        the last valid transition of the episode. The data on index i+1 describe
        the first transition in the following episode.

        If `bootstrap_trunc` is true and an episode is truncated at timestep i,
        gamma * policy.value(next_state) is added to rewards[i]. Note that if you
        are not utilizing a critic network, this should be turned off.

    You can modify this function as you see fit or even replace it entirely.

    """


    states, actions, rewards, dones = [], [], [], []
    steps = 0
    
    while steps < step_limit:

            obs, _ = env.reset()
            obs = tu.to_torch(obs)

            done = False

            while not done:

                # Remember to cast observations to tensors for your models
                action = policy.play(obs)

                states.append(obs)
                actions.append(action)

                obs, reward, terminated, truncated, _ = env.step(action)

                steps += 1
                obs = tu.to_torch(obs)

                truncated = truncated or steps == step_limit
        
                # Optionally bootstrap on truncation
                if truncated and bootstrap_trunc:
                    bootstrap = tu.to_numpy(gamma * policy.value_no_grad(obs))[0]
                    reward += bootstrap

                rewards.append(reward)

                if terminated or truncated:
                    done = True 

                dones.append(done)

    return states, actions, rewards, dones
    


"""
    Trainer - this time you can modify, or even delete the whole class.
    The only required interface are the four functions at the end of this file
    that train your Policy on the Cartpole/Acrobot/Lunar Lander/Car Racing environments.

    This class should give you an idea what you need to implement and the
    hyperparameters that you need to consider.
"""
class PGTrainer(Trainer):

    def __init__(self, env, state_dim, num_actions, 
                 policy_lr=1e-3, value_lr=1e-3,
                 gae_lambda=0.99, batch_size=10000):
        """
            env: The environment to train on
            state_dim: The dimension of the state space
            num_actions: The number of actions in the action space
            policy_lr: The learning rate for the policy network.
            value_lr: The learning rate for the value network. 
            gae_lambda: The GAE discounting parameter lambda
            batch_size: The batch size (num of steps from env for each
            learning iteration)
        """

        self.env = env
        self.batch_size = batch_size

        self.policy_net = PolicyNet(state_dim, num_actions)
        self.value_net = ValueNet(state_dim, 1)

        # Optimizers for each of the nets
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(),
                lr=policy_lr)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(),
                lr=value_lr)

        # TODO: Initialize the remaining parameters


    def train(self, gamma, train_steps) -> PGPolicy:

        """
            Train the agent for number of steps specified by `train_steps`, 
            while using the supplied discount `gamma`.

            Training will proceed by sampling batches of episodes 
            using `collect_trajectories` and constructing the appropriate
            loss function.
        """

        learning_steps = train_steps // self.batch_size
        self.env.reset()

        for i in range(learning_steps):

            policy = PGPolicy(self.policy_net, self.value_net)

            states, actions, rewards, dones = collect_trajectories(self.env, policy, self.batch_size, gamma, bootstrap_trunc=False)

            # Feed this to your neworks
            state_tensor = torch.stack(states)
            action_tensor = torch.tensor(actions)
            ...

            # Get returns and/or advantages for the loss...
            self.calculate_returns(rewards, dones, gamma)
            self.calculate_gae(rewards, state_tensor, dones, gamma)

            # Update the networks and repeat
            self.update(state_tensor, action_tensor, advantages, returns)


        return PGPolicy(self.policy_net, self.value_net)


    def calculate_returns(self, rewards, dones, gamma):

        """
            For each collected timestep in the environment, calculate the
            discounted return from that point to the end of episode
        """

        res = torch.zeros(len(rewards))

        for i in range(len(rewards) - 1, -1, -1):
            # Calculate discounted returns..
            pass

        return res


    def calculate_gae(self, rewards, states, dones, gamma):
        """
            For each collected timestep in the environment, calculate the 
            Generalized Advantage Estimate.
        """

        res = torch.zeros(len(rewards))

        # Get the time lagged values
        values = self.value_net.value_no_grad(states)

        # Calculate GAE for each timestep
        for i in range(len(rewards) - 1, -1, -1):
            # Calculate GAE
            pass

        return res


    def update(self, states, actions, advantages, returns):

        # Zero the gradients
        self.value_optimizer.zero_grad()
        self.policy_optimizer.zero_grad()

        # Calculate values and log probabilites under the current networks (these should be differentiable)
        values = ...
        logprobs = ...

        # Construct the loss and take a learning step
        ...

        self.policy_optimizer.step()
        self.value_optimizer.step()


def get_env_dimensions(env):
    """
        Helper function to get dimensions of state/action spaces of gym environments.
    """

    def get_space_dimensions(space):
        if isinstance(space, gym.spaces.Discrete):
            return space.n
        elif isinstance(space, gym.spaces.Box):
            return np.prod(space.shape)
        else:
            raise TypeError(f"Space type {type(space)} in get_dimensions not recognized, not an instance of Discrete/Box")

    state_dim = get_space_dimensions(env.observation_space)
    num_actions = get_space_dimensions(env.action_space)

    return state_dim, num_actions


"""
    The following four functions will be used to train your agents on the
    respective environments.

    You can use different hyperparameters for each task, just make sure you
    return an object extending the policy interface (i.e one that can `play()`
    actions) so we can evaluate and compare your solutions.
"""

def train_cartpole(env, train_steps, gamma) -> PGPolicy:
    pass

def train_acrobot(env, train_steps, gamma) -> PGPolicy:
    pass

def train_lunarlander(env, train_steps, gamma) -> PGPolicy:
    pass


"""
    CarRacing is a challenging environment for you to try to solve.
"""

RACING_CONTINUOUS = False


def train_carracing(env, train_steps, gamma) -> PGPolicy:
    """
        As the observations are 96x96 RGB images you can either use a
        convolutional neural network, or you have to flatten the observations.

        You can use gymnasium wrappers to achieve the second goal:
    """
    env = gym.wrappers.FlattenObservation(env)

    """
        The episodes in this environment can be very long, you can also limit
        their length by using another wrapper.

        Wrappers can be applied sequentially like so:
    """

    env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)

    human_env = gym.wrappers.FlattenObservation(gym.make("CarRacing-v2",
                                                        continuous=RACING_CONTINUOUS,
                                                        render_mode="human"))


    # Training example
    states, num_actions = get_env_dimensions(env)
    trainer = PGTrainer(env, states, num_actions)
    policy = trainer.train(0.99, train_steps)


    # Run on rendered environment
    obs, _ = human_env.reset()

    obs = tu.to_torch(obs)

    for i in range(200):
        # Go forward
        obs, reward, trunc, term, _ = human_env.step(3)



def wrap_carracing(env):
    """
       Preprocess the environment in any way you want using wrappers.

       This will be used to prepare the evaluation environments for your
       implementation, so you should use the same preprocessing here as you did
       for the training.

       Either use the wrappers offered by gym, or your own, but make sure that
       yours the required `step()` and `reset()` interface.

       For example:
        env = gym.wrappers.FlattenObservation(env)
        return env

        etc.
    """
    return env

def wrap_cartpole(env):
    return env

def wrap_acrobot(env):
    return env

def wrap_lunarlander(env):
    return env




if __name__ == "__main__":
    """
        The flag RACING_CONTINUOUS determines whether the CarRacing environment
        should use a continuous action space. Set it to True if you want to
        experiment with a continuous action space. The evaluation will be done
        based on the value of this flag.
    """
    env = gym.make("CarRacing-v2", continuous=RACING_CONTINUOUS)
    train_carracing(env, 1000, 0.99)

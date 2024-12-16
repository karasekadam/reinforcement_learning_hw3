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
        self.fc1 = nn.Linear(input_size, hidden_size)  # First fully connected layer
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # Second fully connected layer
        self.fc_out = nn.Linear(hidden_size, output_size)  # Output layer

        # Implement the network architecture, see torch.nn layers.
        
    def forward(self, x):
        # Add activation functions and such
        x = F.relu(self.fc1(x))  # First hidden layer with ReLU activation
        x = F.relu(self.fc2(x))  # Second hidden layer with ReLU activation
        x = self.fc_out(x)  # Output layer (no activation for value prediction)
        return x

    @torch.no_grad()
    def value_no_grad(self, obs):
        return self(obs)

    def value(self, obs):
        return self(obs)


class PolicyNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

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
    
    @torch.no_grad()
    def value_no_grad(self, state):
        """
        Evaluate the value of a state without computing gradients.
        """
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
        self.gae_lambda = gae_lambda

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
        total_steps = 0

        while total_steps < train_steps:
            if total_steps != 0 and total_steps % 1000:
                print(f"Currently on step {total_steps} [{total_steps/train_steps * 100 :.2f}%]")
            policy = PGPolicy(self.policy_net, self.value_net)
            states, actions, rewards, dones = collect_trajectories(
                self.env, policy, self.batch_size, gamma, bootstrap_trunc=True
            )

            # Convert to tensors
            state_tensor = torch.stack(states).requires_grad_()
            action_tensor = torch.tensor(actions, dtype=torch.long)
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32)

            # Compute returns and advantages
            returns = self.calculate_returns(rewards_tensor, dones, gamma)
            advantages = self.calculate_gae(rewards_tensor, state_tensor, dones, gamma)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # Normalize

            # Update the networks
            self.update(state_tensor, action_tensor, advantages, returns)

            total_steps += len(rewards)

        return PGPolicy(self.policy_net, self.value_net)

    #     learning_steps = train_steps // self.batch_size
    #     self.env.reset()

    #     for i in range(learning_steps):

    #         policy = PGPolicy(self.policy_net, self.value_net)

    #         states, actions, rewards, dones = collect_trajectories(self.env, policy, self.batch_size, gamma, bootstrap_trunc=False)

    #         # Feed this to your neworks
    #         state_tensor = torch.stack(states)
    #         action_tensor = torch.tensor(actions)
    #         ...

    #         # Get returns and/or advantages for the loss...
    #         self.calculate_returns(rewards, dones, gamma)
    #         self.calculate_gae(rewards, state_tensor, dones, gamma)

    #         # Update the networks and repeat
    #         self.update(state_tensor, action_tensor, advantages, returns)


    #     return PGPolicy(self.policy_net, self.value_net)


    def calculate_returns(self, rewards, dones, gamma):

        """
            For each collected timestep in the environment, calculate the
            discounted return from that point to the end of episode
        """
        
        res = torch.zeros_like(rewards)
        cumul_reward = 0  # This variable will hold the cumulative reward

        for i in range(len(rewards) - 1, -1, -1) :
            if dones[i]:
                cumul_reward = 0
            cumul_reward = res[i] + gamma * cumul_reward
            res[i] = cumul_reward
        return res


    def calculate_gae(self, rewards, states, dones, gamma):
        """
            For each collected timestep in the environment, calculate the 
            Generalized Advantage Estimate.
        """

        res = torch.zeros(len(rewards))

        # Get the time lagged values
        values = self.value_net.value_no_grad(states)
        gae = 0

        # Calculate GAE for each timestep
        for i in range(len(rewards) - 1, -1, -1):
            if dones[i]:
                gae = 0  # Reset GAE at episode boundaries
            delta = rewards[i] + gamma * (values[i + 1] if i + 1 < len(values) else 0) - values[i]
            gae = delta + gamma * self.gae_lambda * gae
            res[i] = gae

        return res


    def update(self, states, actions, advantages, returns):

        # Zero the gradients
        self.value_optimizer.zero_grad()
        self.policy_optimizer.zero_grad()

        # Calculate values and log probabilites under the current networks (these should be differentiable)
        values = self.value_net(states).squeeze()
        log_probs = self.policy_net.log_probs(states, actions)

        advantages = advantages.detach()

        # Construct the loss and take a learning step
        policy_loss = -(log_probs * advantages).mean()
        value_loss = ((returns - values) ** 2).mean()

        policy_loss.backward()
        value_loss.backward()

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
    """
    Train a policy on the CartPole-v1 environment.

    Args:
        env: The CartPole-v1 environment.
        train_steps: Total number of environment steps to train.
        gamma: Discount factor for rewards.

    Returns:
        A trained policy implementing the PGPolicy interface.
    """

    # Wrapping the environment (if needed)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=500)

    # Get dimensions of the environment's state and action spaces
    state_dim, num_actions = get_env_dimensions(env)

    # Initialize the PGTrainer
    trainer = PGTrainer(
        env=env,
        state_dim=state_dim,
        num_actions=num_actions,
        policy_lr=1e-3,  # Learning rate for the policy network
        value_lr=1e-3,   # Learning rate for the value network
        gae_lambda=0.95, # GAE lambda parameter
        batch_size=500   # Batch size (number of steps per update)
    )

    # Train the policy
    trained_policy = trainer.train(gamma=gamma, train_steps=train_steps)

    # Render the trained policy on a human-readable environment
    human_env = gym.wrappers.TimeLimit(gym.make("CartPole-v1", render_mode="human"), max_episode_steps=500)
    obs, _ = human_env.reset()

    total_reward = 0
    done = False

    while not done:
        # Convert observation to tensor
        obs = tu.to_torch(obs)

        # Play an action using the trained policy
        action = trained_policy.play(obs)

        # Take the action in the environment
        obs, reward, done, _, _ = human_env.step(action)
        total_reward += reward

    print("Total Reward with Trained Policy:", total_reward)

    return trained_policy

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
    # env = gym.make("CartPole-v1", continuous=RACING_CONTINUOUS)
    env = gym.make("CartPole-v1")
    # train_carracing(env, 1000, 0.99)
    train_cartpole(env, 5000, 0.25)

# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This example provides an overview of how to implement simple reinforcement
learning algorithms using JAX good practices.

More precisely, we are going to implement a Q-Learning modification called 
Deep Q-Learning. The algorithm is described in the paper 
Playing Atari with Deep Reinforcement Learning.

To demonstrate that the follwing implementation works, we provide and end to 
end interaction of a Deep Q-Learning agent with the gym environment CartPole.
At the OpenAI gym webpage, CartPole environment is described as follows:

> A pole is attached by an un-actuated joint to a cart, which moves along a 
frictionless track. The system is controlled by applying a force 
of +1 or -1 to the cart.

Requirements
------------
To run this script you will need an extra requirement. To install it run

$ pip install gym

References
----------

- Playing Atari with Deep Reinforcement Learning https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
- OpenAI gym https://gym.openai.com/

"""
import random
from typing import Callable, Mapping, NamedTuple, Tuple, Sequence
from functools import partial

import gym # RL environments
import jax # Autograd package
import jax.numpy as np # GPU NumPy :)

import numpy as onp

import matplotlib.pyplot as plt

from collections import deque # Cyclic list with max capacity

# Number of max episodes to train the agent
MAX_EPISODES = 300
# Number of episodes that will train
TRAINING_EPISODES = 220

# We consider the agent intelligent enough when it is capable of holding the 
# pole for at least 100 timesteps
MAX_EPISODE_STEPS = 100

# Number of instances used to train at every timestep with experience replay
BATCH_SIZE = 32
# Discount factor for future rewards
GAMMA = .99
# Noise to perform epsilon greedy exploration
EPSILON = .3

# A transition or a experience is a "recorded" interaction of the agent
# with the environment. A Transition contains:
#  - state: An observation of the environment
#  - next_state: The environment observation after taking an action
#  - action: The decision took by the agent at the transition timestep
#  - reward: The reported benefit of taking the specified action
#  - is_terminal: Whether or not the observation is a "game over"
class Transition(NamedTuple):
    state: np.ndarray
    next_state: np.ndarray
    action: int
    reward: int
    is_terminal: bool

# Types for declaring the neural networks functions simplier and
# more intuitive
ActivationFn = Callable[[np.ndarray], np.ndarray]
Parameters = Mapping[str, np.ndarray]
InitFn = Callable[[np.ndarray], Parameters]
ForwardFn = Callable[[Parameters, np.ndarray], np.ndarray]
Layer = Tuple[InitFn, ForwardFn]

# Create a linear layer (If you come of a Keras background, a dense layer)
def linear(in_features: int, 
           out_features: int, 
           activation: ActivationFn = lambda x: x) -> Layer:

    def init_fn(random_key: np.ndarray) -> Parameters:
        W_key, b_key = jax.random.split(random_key)

        x_init = jax.nn.initializers.xavier_uniform() 
        norm_init = jax.nn.initializers.normal()

        W = x_init(W_key, shape=(out_features, in_features))
        bias = norm_init(b_key, shape=())
        return dict(W=W, bias=bias)

    def forward_fn(params: Parameters, x: np.ndarray) -> np.ndarray:
        W = params['W']
        bias = params['bias']
        return activation(np.dot(W, x) + bias)
    
    return init_fn, forward_fn

# Creates a sequential model that simply reduces the input over a sequence
# of layers
def sequential(*layers: Sequence[Layer]):
    init_fns, forward_fns = zip(*layers)

    def init_fn(random_key: np.ndarray):
        layer_keys = jax.random.split(random_key, num=len(layers))
        return [init_fn(k) for init_fn, k in zip(init_fns, layer_keys)]
    
    def forward_fn(params: Sequence[Parameters], x: np.ndarray) -> np.ndarray:
        for fn, p in zip(forward_fns, params):
            x = fn(p, x)
        return x

    return init_fn, forward_fn

# Simple optimizer that applies the following formula to update the parameters:
# param' = param - learning_rate * grad
def sgd(params: Sequence[Parameters], 
        gradients: Sequence[Parameters],
        learning_rate: float) -> Sequence[Parameters]:
    new_parameters = []
    for i in range(len(params)):
        zipped_grads = zip(params[i].items(), gradients[i].items())
        new_parameters.append({
            k: v - dv * learning_rate for (k, v), (_, dv) in zipped_grads})

    return new_parameters

key = jax.random.PRNGKey(0)

# Create the deep QNetwork
dqn_init_fn, dqn_forward_fn = sequential(
    linear(4, 32, jax.nn.relu),
    linear(32, 32, jax.nn.relu),
    linear(32, 2))

key, subkey = jax.random.split(key)
dqn_parameters = dqn_init_fn(subkey)

# Vectorize the model to work with batches
dqn_forward_fn = jax.vmap(dqn_forward_fn, in_axes=(None, 0))

# We create a copy of parameters to compute the target QValues 
target_parameters = dqn_parameters

# Mean squared error as loss function
mse = lambda y1, y2: (y1 - y2) ** 2

@jax.grad # Differentiate the loss with respect to the model weights
def compute_loss(parameters, x, y, actions):
    # Get the q values corresponding to specified actions
    q_values = dqn_forward_fn(parameters, x)
    q_values = q_values[np.arange(x.shape[0]), actions]
    return np.mean(mse(y, q_values))

# We compile the function with jit to improve performance
backward_fn = jax.jit(compute_loss)

# Declare an SGD optimizer
optimizer = partial(sgd, learning_rate=1e-3)
optimizer = jax.jit(optimizer)


# Take action
def take_action(key, state):
    key, sk = jax.random.split(key)
    # We are using an epsilon greedy policy for exploration
    # Meaning that at every timestep we flip a biased coin with EPSILON 
    # probability of being true and 1- EPSILON of being false. In case it is true
    # we take a random action
    if jax.random.uniform(sk, shape=(1,)) < EPSILON:
        # Remember that we have only 2 actions (left, right)
        action = jax.random.randint(sk, shape=(1,), minval=0, maxval=2)
    else:
        state = np.expand_dims(state, axis=0)
        q_values = dqn_forward_fn(dqn_parameters, state)[0]
        # Pick the action that maximizes the value
        action = np.argmax(q_values)
    
    return int(action)


# Train using experience replay
def train():
    if len(memory) < BATCH_SIZE:
        return dqn_parameters # No train because we do not have enough experiences
    # Experience replay
    transitions = random.sample(memory, k=BATCH_SIZE)
        
    # Convert transition into tensors. Converting list of tuples to
    # tuple of lists
    transitions = Transition(*zip(*transitions))
    states = np.array(transitions.state)
    next_states = np.array(transitions.next_state)
    actions = np.array(transitions.action)
    rewards = np.array(transitions.reward)
    is_terminal = np.array(transitions.is_terminal)

    # Compute the next Q values using the target parameters
    # We vectorize the model using vmap to work with batches
    next_Q_values = dqn_forward_fn(target_parameters, next_states)
    # Bellman equation
    yj = rewards + GAMMA * np.max(next_Q_values, axis=-1)
    # In case of terminal state we set a 0 reward
    yj = np.where(is_terminal, 0, yj)

    # Compute the Qvalues corresponding to the sampled transitions
    # and backpropagate the mse loss gradients
    gradients = backward_fn(dqn_parameters, states, yj, actions)
    
    # Update the deep q network parameters using our optimizer
    return optimizer(dqn_parameters, gradients)
    

env = gym.make('CartPole-v1')

# At maximum store 1000 transitions 
# Usually the replay memory buffer should be larger, but with simple 
# environments, as
memory = deque(maxlen=int(1e3))

# Moving average for result reporting
ma = lambda x, w: onp.convolve(x, onp.ones(w), 'valid') / w

timesteps_history = []

for episode in range(MAX_EPISODES):
    state = env.reset()
    training = episode < TRAINING_EPISODES
    if not training:
        EPSILON = 0 # No more random actions

    for timestep in range(MAX_EPISODE_STEPS):
        if not training:
            # Only render the environment after training
            env.render() 
        
        # Take an action
        # With jax is mandatory to provide a random key at any random operation
        key, subkey = jax.random.split(key)
        action = take_action(subkey, state)
        next_state, reward, done, _ = env.step(action)
        
        # Generate a transition
        t = Transition(state=state, next_state=next_state, 
                       reward=reward, is_terminal=done,
                       action=action)
        memory.append(t) # Store the transition
        if training:
            dqn_parameters = train() # Update the agent with experience replay
        
        # Every timestep we reduce the probability of taking random actions
        # this is because we want to explotate what we have learn during the 
        # previous episodes
        EPSILON -= 2e-5
        
        state = next_state

        if done: 
            break # The pole has felt, we should start a new episode
    
    print(f'Episode[{episode}]: Agent held the pole for {timestep} timesteps')
    
    # Plot the timesteps of every episode to see the improvements
    timesteps_history.append(timestep )
    if len(timesteps_history) > 5:
    
        avg_timesteps = ma(timesteps_history, 5)
        plt.plot(range(len(avg_timesteps)), avg_timesteps)
        plt.xlabel('Episodes')
        plt.ylabel('Timesteps')
        plt.title('Timesteps over episodes')
        plt.savefig('report.png')

    # At the end of the episode we update the target parameters
    target_parameters = dqn_parameters

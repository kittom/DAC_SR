import pickle
import gzip

import scipy.signal
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from itertools import count
from collections import namedtuple
import time
import sys
import os
import json

sys.path.append(os.path.dirname(__file__))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from leadingones_eval import LeadingOnesEval


def tt(ndarray):
    """
    Helper Function to cast observation to correct type/device
    """
    if device == "cuda":
        return Variable(torch.from_numpy(ndarray).float().cuda(), requires_grad=False)
    else:
        return Variable(torch.from_numpy(ndarray).float(), requires_grad=False)


def soft_update(target, source, tau):
    """
    Simple Helper for updating target-network parameters
    :param target: target network
    :param source: policy network
    :param tau: weight to regulate how strongly to update (1 -> copy over weights)
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    """
    See soft_update
    """
    soft_update(target, source, 1.0)


class Q(nn.Module):
    """
    Simple fully connected Q function. Also used for skip-Q when concatenating behaviour action and state together.
    Used for simpler environments such as mountain-car or lunar-lander.
    """

    def __init__(self, state_dim, action_dim, non_linearity=F.relu, hidden_dim=50):
        super(Q, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self._non_linearity = non_linearity

    def forward(self, x):
        x = self._non_linearity(self.fc1(x))
        x = self._non_linearity(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    """
    Simple Replay Buffer. Used for standard DQN learning.
    """

    def __init__(self, max_size, rng = np.random.default_rng()):
        self._data = namedtuple(
            "ReplayBuffer",
            ["states", "actions", "next_states", "rewards", "terminal_flags"],
        )
        self._data = self._data(
            states=[], actions=[], next_states=[], rewards=[], terminal_flags=[]
        )
        self._size = 0
        self._max_size = max_size
        self.rng = rng

    def add_transition(self, state, action, next_state, reward, done):
        self._data.states.append(state)
        self._data.actions.append(action)
        self._data.next_states.append(next_state)
        self._data.rewards.append(reward)
        self._data.terminal_flags.append(done)
        self._size += 1

        if self._size > self._max_size:
            self._data.states.pop(0)
            self._data.actions.pop(0)
            self._data.next_states.pop(0)
            self._data.rewards.pop(0)
            self._data.terminal_flags.pop(0)

    def random_next_batch(self, batch_size):
        batch_indices = self.rng.choice(len(self._data.states), batch_size)
        batch_states = np.array([self._data.states[i] for i in batch_indices])
        batch_actions = np.array([self._data.actions[i] for i in batch_indices])
        batch_next_states = np.array([self._data.next_states[i] for i in batch_indices])
        batch_rewards = np.array([self._data.rewards[i] for i in batch_indices])
        batch_terminal_flags = np.array(
            [self._data.terminal_flags[i] for i in batch_indices]
        )
        return (
            tt(batch_states),
            tt(batch_actions),
            tt(batch_next_states),
            tt(batch_rewards),
            tt(batch_terminal_flags),
        )

    def save(self, path):
        with open(os.path.join(path, "rpb.pkl"), "wb") as fh:
            pickle.dump(list(self._data), fh)

    def load(self, path):
        with open(os.path.join(path, "rpb.pkl"), "rb") as fh:
            data = pickle.load(fh)
        self._data = namedtuple(
            "ReplayBuffer",
            ["states", "actions", "next_states", "rewards", "terminal_flags"],
        )
        self._data.states = data[0]
        self._data.actions = data[1]
        self._data.next_states = data[2]
        self._data.rewards = data[3]
        self._data.terminal_flags = data[4]
        self._size = len(data[0])


class DQN:
    """
    Simple double DQN Agent
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        env: gym.Env,
        eval_env: gym.Env = None,
        extra_eval_env: gym.Env = None,
        vision: bool = False,
        out_dir: str = "./output/",
        gamma: float = 0.99,
        loss_function = F.mse_loss,
        use_double_dqn = True,
        seed: int = 42,
    ):

        """
        Initialize the DQN Agent
        :param state_dim: dimensionality of the input states
        :param action_dim: dimensionality of the output actions
        :param gamma: discount factor
        :param env: environment to train on
        :param eval_env: environment to evaluate on
        :param extra_eval_env: another environment to evaluate on (optional)
        :param vision: boolean flag to indicate if the input state is an image or not
        """
        if not vision:  # For featurized states
            self._q = Q(state_dim, action_dim).to(device)
            self._q_target = Q(state_dim, action_dim).to(device)
        else:  # For image states, i.e. Atari
            raise NotImplementedError
        
        # set random seed
        self.rng = np.random.default_rng(seed=seed)
        torch.manual_seed(seed)

        self._eval_r_best = None

        self._gamma = gamma
        self._loss_function = loss_function
        self._q_optimizer = optim.Adam(self._q.parameters(), lr=0.001)
        self._action_dim = action_dim
        self.use_double_dqn = use_double_dqn

        self._replay_buffer = ReplayBuffer(1e6, self.rng)
        self._env = env
        self._eval_env = eval_env
        self._extra_eval_env = extra_eval_env
        self.out_dir = out_dir

        if self.out_dir is not None:
            os.makedirs(self.out_dir, exist_ok=True)

    def save_rpb(self, path):
        self._replay_buffer.save(path)

    def load_rpb(self, path):
        self._replay_buffer.load(path)

    def act(self, x: np.ndarray, epsilon: float) -> int:
        """
        Simple helper to get action epsilon-greedy based on observation x
        """
        u = np.argmax(self._q(tt(x)).detach().numpy())
        r = self.rng.random()
        if r < epsilon:
            return self.rng.integers(low=0, high=self._action_dim)
        return u
    
    def update_epsilon(self, begin_learning_after, max_train_time_steps, epsilon_start, epsilon_decay_end_point, epsilon_decay_end_value, cur_total_steps) -> float:
        """
        Calculate new epsilon value (linearly decay)
        """
        # the training step when epsilon is set to constant (as epsilon_decay_end_value)
        end_step = begin_learning_after + (max_train_time_steps - begin_learning_after) * epsilon_decay_end_point

        if cur_total_steps >= end_step:
            return epsilon_decay_end_value
        
        return epsilon_start - (epsilon_start - epsilon_decay_end_value) * (cur_total_steps - begin_learning_after) / (end_step - begin_learning_after)

    def train(
        self,
        episodes: int,
        max_env_time_steps: int = 1_000_000,
        epsilon: float = 0.2,
        epsilon_decay: bool = False,
        epsilon_decay_end_point: float = 0.5, # ignored if epsilon_decay = False
        epsilon_decay_end_value: float = 0.05, # ignored if epsilon_decay = False
        eval_every_n_steps: int = 1000,
        n_eval_episodes_per_instance: int = 20,
        save_agent_at_every_eval: bool = True,
        max_train_time_steps: int = 1_000_000,
        begin_learning_after: int = 10_000,
        batch_size: int = 2_048,
        log_level=1,
        use_formula=True
    ):
        """
        Training loop
        :param episodes: maximum number of episodes to train for
        :param max_env_time_steps: maximum number of steps in the environment to perform per episode
        :param epsilon: constant epsilon for exploration when selecting actions
        :param eval_every_n_steps: interval of steps after which to evaluate the trained agent
        :param max_train_time_steps: maximum number of steps to train
        :param epsilon_decay: whether epsilon decay is enabled. ONLY WORK IF TRAINING BUDGET IS BASED ON max_train_time_steps.
        :param log_level:
            (1) basic log: evaluation scores and evaluation episodes' infos
            (2) extensive log: (1) and discounted rewards per episode (evaluation), average value-action estimate per episode (evaluation), average discounted rewards per episode (evaluation), average loss per batch
        :return:
        """

        total_steps = 0
        start_time = time.time()

        self.evaluator = LeadingOnesEval(eval_env=self._eval_env,
                                         agent=self,
                                         use_formula=use_formula,
                                         n_eval_episodes_per_instance=n_eval_episodes_per_instance,
                                         log_path=os.path.join(self.out_dir, "eval"),
                                         save_agent_at_every_eval=save_agent_at_every_eval,
                                         verbose=log_level,
                                         name = "eval")

        if self._extra_eval_env:
            self.extra_evaluator = LeadingOnesEval(eval_env=self._extra_eval_env,
                                         agent=self,
                                         use_formula=use_formula,
                                         n_eval_episodes_per_instance=n_eval_episodes_per_instance,
                                         log_path=os.path.join(self.out_dir, "extra_eval"),
                                         save_agent_at_every_eval=False,    # we already save the agent via self.evaluator (if needed), so no need to re-save it here
                                         verbose=log_level,
                                         name = "extra")

        s = self._env.get_state()
        epsilon_start = epsilon
        for e in range(episodes):
            ep_loss = []
            for t in range(max_env_time_steps):
                
                if epsilon_decay:
                    epsilon = self.update_epsilon(begin_learning_after, max_train_time_steps, epsilon_start, epsilon_decay_end_point, epsilon_decay_end_value, total_steps)

                #print(f"epsilon: {epsilon:.5f}")
                a = self.act(
                    s, epsilon if total_steps > begin_learning_after else 1.0
                )
                ns, r, tr, d, _ = self._env.step(a)
                total_steps += 1
                #print("#episodes=%d, #steps=%d" % (e + 1, total_steps), end="\r")

                if (total_steps % eval_every_n_steps) == 0:
                    self.evaluator.eval(total_steps)
                    if self._extra_eval_env:
                        self._extra_eval_env(total_steps)
    
                # Update replay buffer
                self._replay_buffer.add_transition(s, a, ns, r, d)
                if begin_learning_after < total_steps:
                    (
                        batch_states,
                        batch_actions,
                        batch_next_states,
                        batch_rewards,
                        batch_terminal_flags,
                    ) = self._replay_buffer.random_next_batch(batch_size)

                    ########### Begin Q-learning update
                    if self.use_double_dqn:
                        target = (
                            batch_rewards
                            + (1 - batch_terminal_flags)
                            * self._gamma
                            * self._q_target(batch_next_states)[
                                torch.arange(batch_size).long(),
                                torch.argmax(self._q(batch_next_states), dim=1),
                            ]
                        )
                    else:
                        target = (
                            batch_rewards
                            + (1 - batch_terminal_flags)
                            * self._gamma
                            * self._q_target(batch_next_states).max(dim=1).values
                        )
                        
                    current_prediction = self._q(batch_states)[
                        torch.arange(batch_size).long(), batch_actions.long()
                    ]

                    loss = self._loss_function(current_prediction, target.detach())
                    if log_level > 1:
                        ep_loss.append(loss.detach().numpy())

                    self._q_optimizer.zero_grad()
                    loss.backward()
                    self._q_optimizer.step()

                    soft_update(self._q_target, self._q, 0.01)
                    ########### End double Q-learning update

                if d:
                    break
                s = ns
                if total_steps >= max_train_time_steps:
                    break
            
            s, _ = self._env.reset()
            
            if total_steps >= max_train_time_steps:
                break

        best_model_path = os.path.join(os.path.join(self.out_dir, "eval"), "best_model.pt")
        return best_model_path
        

    def __repr__(self):
        return "DDQN"


    def save_model(self, model_path):
        torch.save(self._q.state_dict(), os.path.join(model_path + ".pt"))

    def load(self, path):
        self._q.load_state_dict(torch.load(path))

    def get_actions_for_all_states(self, n):
        acts = [
            self.act(np.array([n, fx]),0)
            for fx in range(0, n)
        ]
        #return [self._env.action_choices[self._env.inst_id][act] for act in acts]
        return acts


# class DQNAgent:
#     def __init__(self, env, model, name="TheoryDQNPolicy"):
#         """
#         model: trained model
#         """
#         self.name = name
#         self.env = env
#         state_dim = env.observation_space.shape[0]
#         action_dim = env.action_space.n
#         self.agent = DQN(state_dim, action_dim, env=env, eval_env=env, out_dir="./")
#         self.agent.load(model)

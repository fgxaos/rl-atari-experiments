### LIBRARIES ###
# Global libraries
import os
import copy
from collections import namedtuple
from itertools import count
import math
import random
import time
import wandb

import matplotlib.pyplot as plt

import numpy as np

import gym

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T


# Custom libraries
from utils.wrappers import make_env
from utils.memory import ReplayMemory
from models.dqn.dqn import DQN

### TYPE DEFINITION ###
Transition = namedtuple("Transion", ("state", "action", "next_state", "reward"))


### CLASS DEFINITION ###
class DQNAgent(object):
    def __init__(
        self,
        env,
        n_episodes,
        device,
        batch_size,
        lr,
        gamma,
        eps_start,
        eps_end,
        eps_decay,
        target_update,
        train_render,
        test_render,
        initial_memory,
    ):
        self.device = device
        self.n_episodes = n_episodes
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.target_update = target_update
        self.initial_memory = initial_memory
        self.memory_size = 10 * initial_memory
        self.train_render = train_render
        self.test_render = test_render
        self.model_name = f"dqn_{env}_model"

        env = gym.make(env)
        self.env = make_env(env)
        self.n_actions = self.env.action_space.n

        # Create the networks
        input_dim = 4
        self.policy_net = DQN(input_dim, self.n_actions).to(self.device)
        self.target_net = DQN(input_dim, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Setup the optimizer
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)

        self.steps_done = 0

        # Initialize the replay memory
        self.memory = ReplayMemory(self.memory_size)

        # Train the model
        train_rewards = self.train()
        plt.plot(train_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Training rewards")
        plt.show()
        # Save the model
        torch.save(self.policy_net, self.model_name)
        self.policy_net = torch.load(self.model_name)
        self.test()

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -1.0 * self.steps_done / self.eps_decay
        )
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state.to(self.device)).max(1)[1].view(1, 1)
        else:
            return torch.tensor(
                [[random.randrange(self.n_actions)]],
                device=self.device,
                dtype=torch.long,
            )

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        actions = tuple(
            (map(lambda a: torch.tensor([[a]], device=self.device), batch.action))
        )
        rewards = tuple(
            (map(lambda r: torch.tensor([r], device=self.device), batch.reward))
        )

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        ).to(self.device)

        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.cat(actions)
        reward_batch = torch.cat(rewards)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = (
            self.target_net(non_final_next_states).max(1)[0].detach()
        )
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        loss = F.smooth_l1_loss(
            state_action_values, expected_state_action_values.unsqueeze(1)
        )
        wandb.log({"train_loss": loss})

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def get_state(self, obs):
        state = np.array(obs)
        state = state.transpose((2, 0, 1))
        state = torch.from_numpy(state)
        return state.unsqueeze(0)

    def train(self):
        train_rewards = []
        for episode in range(self.n_episodes):
            obs = self.env.reset()
            state = self.get_state(obs)
            total_reward = 0.0
            for t in count():
                action = self.select_action(state)
                if self.train_render:
                    self.env.render()
                obs, reward, done, info = self.env.step(action)
                total_reward += reward

                if not done or info["ale.lives"] > 0:
                    next_state = self.get_state(obs)
                else:
                    next_state = None

                reward = torch.tensor([reward], device=self.device)
                self.memory.push(
                    state, action.to(self.device), next_state, reward.to(self.device)
                )
                state = next_state

                if self.steps_done > self.initial_memory:
                    self.optimize_model()

                    if self.steps_done % self.target_update == 0:
                        self.target_net.load_state_dict(self.policy_net.state_dict())

                if done and info["ale.lives"] == 0:
                    train_rewards.append(total_reward)
                    wandb.log(
                        {"reward": total_reward, "reward_step": len(train_rewards)}
                    )
                    break

            if episode % 1 == 0:
                print(
                    f"Steps: {t}({self.steps_done})\tEpisode: {episode+1}/{self.n_episodes}\tTotal reward: {total_reward}"
                )

        self.env.close()
        return train_rewards

    def test(self):
        env = gym.wrappers.Monitor(
            self.env, os.path.join("videos", f"{self.model_name}_video")
        )
        for episode in range(self.n_episodes):
            obs = env.reset()
            state = self.get_state(obs)
            total_reward = 0.0
            for t in count():
                action = self.policy_net(state.to(self.device)).max(1)[1].view(1, 1)
                if self.test_render:
                    env.render()
                obs, reward, done, info = env.step(action)
                total_reward += reward
                if not done:
                    next_state = self.get_state(obs)
                else:
                    next_state = None

                state = next_state

                if done:
                    print(f"Finished episode {episode} with reward {total_reward}")
                    break
        env.close()

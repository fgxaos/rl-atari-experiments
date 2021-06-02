### LIBRARIES ###
# Global libraries
import os
import random
import wandb

import numpy as np

import gym

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

# Custom libraries
from models.mnfdqn.mnfdqn import MNFDQN, initialize_weights
from utils.wrappers import make_env
from utils.replay_buffer import ReplayBuffer

### CLASS DEFINITION ###
class MNFDQNAgent(object):
    def __init__(
        self,
        env,
        n_episodes,
        device,
        batch_size,
        lr,
        discount,
        double_q,
        alpha,
        replay_buffer_size,
        hidden_dim,
        n_hidden,
        n_flows_q,
        n_flows_r,
        model,
        target_update_freq,
        learning_starts,
        learning_freq,
        render,
    ):
        env = gym.make(env)
        self.env = make_env(env)
        self.n_actions = self.env.action_space.n
        self.n_episodes = n_episodes
        self.target_update_freq = target_update_freq
        self.learning_starts = learning_starts

        self.batch_size = batch_size
        self.discount = discount
        self.double_q = double_q
        self.input_dim = self.env.observation_space.shape[-1]
        self.kl_coeff = float(alpha) / replay_buffer_size
        self.learning_freq = learning_freq
        self.render = render
        self.model_name = model

        # Initialize the replay memory
        self.replay_buffer = ReplayBuffer(replay_buffer_size)

        self.online_net = MNFDQN(
            self.input_dim, self.n_actions, hidden_dim, n_hidden, n_flows_q, n_flows_r
        )
        if model and os.path.isfile(model):
            self.online_net.load_state_dict(torch.load(model))
        self.online_net.train()

        self.target_net = MNFDQN(
            self.input_dim, self.n_actions, hidden_dim, n_hidden, n_flows_q, n_flows_r
        )
        self.update_target_net()
        self.target_net.eval()
        for param in self.target_net.parameters():
            param.requires_grad = False

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=lr)

    def update_target_net(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def act(self, state, eval=False):
        # Acts based on single state (no batch)
        if eval:
            self.online_net.eval()
        else:
            self.online_net.train()
        state = torch.FloatTensor(state / 255.0)
        return self.online_net(state, kl=False).data.max(1)[1][0]

    def learn(self, states, actions, rewards, next_states, terminals):
        self.online_net.train()
        self.online_net.reset_noise()
        self.target_net.eval()
        states = torch.FloatTensor(states / 255.0)
        actions = torch.LongTensor(actions)
        next_states = torch.FloatTensor(next_states / 255.0)
        rewards = torch.FloatTensor(rewards).view(-1, 1)
        terminals = torch.FloatTensor(terminals).view(-1, 1)

        # Compute Q(s_t, a)
        # The model computes Q(s_t), then we select the columns of actions taken
        state_values, kl_div = self.online_net(states, kl=True)
        state_action_values = state_values.gather(1, actions.view(-1, 1))

        if self.double_q:
            next_actions = self.online_net(next_states, kl=False).max(1)[1]
            next_state_values = self.target_net(next_states, kl=False).gather(
                1, next_actions.view(-1, 1)
            )
        else:
            next_state_values = self.target_net(next_states, kl=False).max(1)[0]

        # target_state_action_values = rewards + (
        target_state_action_values = abs(rewards) + (
            1 - terminals
        ) * self.discount * next_state_values.view(-1, 1)

        td_errors = F.smooth_l1_loss(
            state_action_values, target_state_action_values.detach(), reduction="mean"
        )

        loss = td_errors + self.kl_coeff * kl_div
        wandb.log({"train_loss": loss})

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.online_net.parameters(), 10)
        self.optimizer.step()

        return td_errors, kl_div, loss

    def train(self):
        obs = self.env.reset()
        self.online_net.reset_noise()
        n_iters = 0
        n_updates = 0
        episode_rewards = [0.0]
        reward_step = 0
        best_score = None
        prev_lives = None

        # Main training loop
        while True:
            n_iters += 1
            # Take action and store transition in the replay buffer
            if n_iters <= self.learning_starts:
                action = random.randrange(self.n_actions)
            else:
                action = self.act(
                    np.transpose(np.array(obs)[None], [0, 3, 1, 2]), eval=False
                )
            if self.render:
                self.env.render()
            new_obs, rew, done, info = self.env.step(int(action))
            death = done or (
                prev_lives is not None
                and info["ale.lives"] < prev_lives
                and info["ale.lives"] > 0
            )
            prev_lives = info["ale.lives"]
            self.replay_buffer.add(obs, action, np.sign(rew), new_obs, float(death))
            obs = new_obs
            episode_rewards[-1] += rew

            if done and info["ale.lives"] == 0:
                print(
                    f"Episode {len(episode_rewards)}: \t Reward {episode_rewards[-1]}"
                )
                wandb.log({"reward": episode_rewards[-1], "reward_step": reward_step})
                reward_step += 1
                episode_rewards.append(0.0)
                obs = self.env.reset()

            if n_iters > self.learning_starts and n_iters % self.learning_freq == 0:
                obses_t, actions, rewards, obses_tp1, dones = self.replay_buffer.sample(
                    self.batch_size
                )
                obses_t = np.transpose(obses_t, [0, 3, 1, 2])
                obses_tp1 = np.transpose(obses_tp1, [0, 3, 1, 2])

                td_errors, kl_reg, loss = self.learn(
                    obses_t, actions, rewards, obses_tp1, dones
                )
                n_updates += 1

            # Update target network
            if (
                n_iters > self.learning_starts
                and n_iters % self.target_update_freq == 0
            ):
                self.update_target_net()

            if len(episode_rewards) > self.n_episodes:
                break

        # Save the model
        torch.save(self.online_net, self.model_name)

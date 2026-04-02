import os

import numpy as np
import torch
import torch.nn as nn

from config import COLS


def get_state(env, player):
    board = env.get_board().astype(np.float32).flatten()
    return board * player


def get_valid_q_values(q_values, legal_actions):
    masked_q_values = q_values.clone()
    illegal_actions = [col for col in range(COLS) if col not in legal_actions]

    for action in illegal_actions:
        masked_q_values[action] = float("-inf")

    return masked_q_values


def mask_batch_q_values(q_values_batch, legal_actions_batch):
    masked_q_values_batch = q_values_batch.clone()

    for row_index, legal_actions in enumerate(legal_actions_batch):
        if not legal_actions:
            continue

        illegal_actions = [col for col in range(COLS) if col not in legal_actions]
        for action in illegal_actions:
            masked_q_values_batch[row_index, action] = float("-inf")

    return masked_q_values_batch


def get_reward(result, player):
    if result == player:
        return 1.0
    if result == 0:
        return 0.0
    if result is None:
        return 0.0
    return -1.0


def dqn_train_step(q_network, target_network, replay_buffer, optimizer, batch_size, gamma, device):
    if len(replay_buffer) < batch_size:
        return None

    states, actions, rewards, next_states, next_legal_actions_batch, dones = replay_buffer.sample(batch_size)

    states_tensor = torch.tensor(states, dtype=torch.float32, device=device)
    actions_tensor = torch.tensor(actions, dtype=torch.int64, device=device).unsqueeze(1)
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
    next_states_tensor = torch.tensor(next_states, dtype=torch.float32, device=device)
    dones_tensor = torch.tensor(dones, dtype=torch.float32, device=device)

    q_network.train()
    current_q_values = q_network(states_tensor).gather(1, actions_tensor).squeeze(1)

    target_network.eval()
    with torch.no_grad():
        next_q_values_batch = target_network(next_states_tensor)
        masked_next_q_values_batch = mask_batch_q_values(next_q_values_batch, next_legal_actions_batch)
        next_q_values = masked_next_q_values_batch.max(dim=1)[0]

        next_q_values = torch.where(
            dones_tensor == 1.0,
            torch.zeros_like(next_q_values),
            next_q_values,
        )

        target_q_values = rewards_tensor + gamma * next_q_values

    loss_fn = nn.SmoothL1Loss()
    loss = loss_fn(current_q_values, target_q_values)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(q_network.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.item()


def double_dqn_train_step(q_network, target_network, replay_buffer, optimizer, batch_size, gamma, device):
    if len(replay_buffer) < batch_size:
        return None

    states, actions, rewards, next_states, next_legal_actions_batch, dones = replay_buffer.sample(batch_size)

    states_tensor = torch.tensor(states, dtype=torch.float32, device=device)
    actions_tensor = torch.tensor(actions, dtype=torch.int64, device=device).unsqueeze(1)
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
    next_states_tensor = torch.tensor(next_states, dtype=torch.float32, device=device)
    dones_tensor = torch.tensor(dones, dtype=torch.float32, device=device)

    q_network.train()
    current_q_values = q_network(states_tensor).gather(1, actions_tensor).squeeze(1)

    with torch.no_grad():
        online_next_q_values_batch = q_network(next_states_tensor)
        masked_online_next_q_values_batch = mask_batch_q_values(
            online_next_q_values_batch,
            next_legal_actions_batch,
        )
        next_actions = torch.argmax(masked_online_next_q_values_batch, dim=1, keepdim=True)

        target_network.eval()
        target_next_q_values_batch = target_network(next_states_tensor)
        next_q_values = target_next_q_values_batch.gather(1, next_actions).squeeze(1)

        next_q_values = torch.where(
            dones_tensor == 1.0,
            torch.zeros_like(next_q_values),
            next_q_values,
        )

        target_q_values = rewards_tensor + gamma * next_q_values

    loss_fn = nn.SmoothL1Loss()
    loss = loss_fn(current_q_values, target_q_values)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(q_network.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.item()


def update_target_network(q_network, target_network):
    target_network.load_state_dict(q_network.state_dict())


def decay_epsilon(agent, epsilon_min, epsilon_decay):
    agent.epsilon = max(epsilon_min, agent.epsilon * epsilon_decay)


def ensure_checkpoint_dir(checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)


def save_checkpoint(
    checkpoint_path,
    episode,
    q_network,
    target_network,
    optimizer,
    replay_buffer,
    epsilon,
    episode_rewards,
    episode_losses,
):
    checkpoint = {
        "episode": episode,
        "q_network_state_dict": q_network.state_dict(),
        "target_network_state_dict": target_network.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "replay_buffer_state_dict": replay_buffer.state_dict(),
        "epsilon": epsilon,
        "episode_rewards": episode_rewards,
        "episode_losses": episode_losses,
    }

    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(
    checkpoint_path,
    q_network,
    target_network,
    optimizer,
    replay_buffer,
    device,
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    q_network.load_state_dict(checkpoint["q_network_state_dict"])
    target_network.load_state_dict(checkpoint["target_network_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    replay_buffer.load_state_dict(checkpoint["replay_buffer_state_dict"])

    start_episode = checkpoint["episode"] + 1
    epsilon = checkpoint["epsilon"]
    episode_rewards = checkpoint.get("episode_rewards", [])
    episode_losses = checkpoint.get("episode_losses", [])

    return start_episode, epsilon, episode_rewards, episode_losses
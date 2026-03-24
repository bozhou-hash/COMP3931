import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import PLAYER_ONE, PLAYER_TWO, COLS
from env.connect4_env import Connect4Env
from agents.random_agent import RandomAgent
from agents.double_dqn_agent import DoubleDQNAgent
from models.q_network import QNetwork
from training.replay_buffer import ReplayBuffer


def get_state(env, player):
    board = env.get_board().astype(np.float32).flatten()
    return board * player


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
        return 0.5
    if result is None:
        return 0.0
    return -1.0


def train_step(q_network, target_network, replay_buffer, optimizer, batch_size, gamma, device):
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
        # Online network selects actions
        online_next_q_values_batch = q_network(next_states_tensor)
        masked_online_next_q_values_batch = mask_batch_q_values(
            online_next_q_values_batch,
            next_legal_actions_batch,
        )
        next_actions = torch.argmax(masked_online_next_q_values_batch, dim=1, keepdim=True)

        # Target network evaluates selected actions
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


def play_training_episode(
    env,
    double_dqn_agent,
    opponent_agent,
    replay_buffer,
    q_network,
    target_network,
    optimizer,
    batch_size,
    gamma,
    device,
):
    env.reset()
    episode_loss_values = []

    while not env.game_over:
        current_player = env.get_current_player()

        if current_player == double_dqn_agent.player:
            state = get_state(env, double_dqn_agent.player)
            action = double_dqn_agent.select_action(env)

            if action is None:
                break

            env.drop_piece(action)
            result = env.get_result()

            if result is not None:
                reward = get_reward(result, double_dqn_agent.player)
                next_state = get_state(env, double_dqn_agent.player)
                next_legal_actions = []
                done = True

                replay_buffer.push(state, action, reward, next_state, next_legal_actions, done)
                loss = train_step(
                    q_network=q_network,
                    target_network=target_network,
                    replay_buffer=replay_buffer,
                    optimizer=optimizer,
                    batch_size=batch_size,
                    gamma=gamma,
                    device=device,
                )
                if loss is not None:
                    episode_loss_values.append(loss)
                break

            opponent_action = opponent_agent.select_action(env)
            if opponent_action is None:
                reward = 0.0
                next_state = get_state(env, double_dqn_agent.player)
                next_legal_actions = []
                done = True

                replay_buffer.push(state, action, reward, next_state, next_legal_actions, done)
                loss = train_step(
                    q_network=q_network,
                    target_network=target_network,
                    replay_buffer=replay_buffer,
                    optimizer=optimizer,
                    batch_size=batch_size,
                    gamma=gamma,
                    device=device,
                )
                if loss is not None:
                    episode_loss_values.append(loss)
                break

            env.drop_piece(opponent_action)
            result = env.get_result()

            reward = get_reward(result, double_dqn_agent.player)
            next_state = get_state(env, double_dqn_agent.player)
            next_legal_actions = env.get_legal_actions() if result is None else []
            done = result is not None

            replay_buffer.push(state, action, reward, next_state, next_legal_actions, done)
            loss = train_step(
                q_network=q_network,
                target_network=target_network,
                replay_buffer=replay_buffer,
                optimizer=optimizer,
                batch_size=batch_size,
                gamma=gamma,
                device=device,
            )
            if loss is not None:
                episode_loss_values.append(loss)

        else:
            action = opponent_agent.select_action(env)
            if action is None:
                break
            env.drop_piece(action)

    if not episode_loss_values:
        return None

    return float(np.mean(episode_loss_values))


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


def train_double_dqn(
    num_episodes=20000,
    batch_size=64,
    gamma=0.99,
    learning_rate=0.0001,
    replay_buffer_capacity=10000,
    target_update_frequency=100,
    epsilon_start=1.0,
    epsilon_min=0.10,
    epsilon_decay=0.9999,
    checkpoint_dir="double_dqn_checkpoints",
    checkpoint_frequency=500,
    resume=False,
    checkpoint_path=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ensure_checkpoint_dir(checkpoint_dir)

    env = Connect4Env()

    q_network = QNetwork().to(device)
    target_network = QNetwork().to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
    replay_buffer = ReplayBuffer(capacity=replay_buffer_capacity)

    update_target_network(q_network, target_network)

    double_dqn_agent = DoubleDQNAgent(
        player=PLAYER_ONE,
        q_network=q_network,
        epsilon=epsilon_start,
        device=device,
    )
    opponent_agent = RandomAgent(player=PLAYER_TWO)

    start_episode = 1
    episode_rewards = []
    episode_losses = []

    latest_checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pth")

    if resume:
        selected_checkpoint_path = checkpoint_path or latest_checkpoint_path

        if os.path.exists(selected_checkpoint_path):
            start_episode, loaded_epsilon, episode_rewards, episode_losses = load_checkpoint(
                checkpoint_path=selected_checkpoint_path,
                q_network=q_network,
                target_network=target_network,
                optimizer=optimizer,
                replay_buffer=replay_buffer,
                device=device,
            )
            double_dqn_agent.epsilon = loaded_epsilon
            print(f"Resumed training from episode {start_episode}")
        else:
            print("No checkpoint found. Starting training from scratch.")

    for episode in range(start_episode, num_episodes + 1):
        episode_loss = play_training_episode(
            env=env,
            double_dqn_agent=double_dqn_agent,
            opponent_agent=opponent_agent,
            replay_buffer=replay_buffer,
            q_network=q_network,
            target_network=target_network,
            optimizer=optimizer,
            batch_size=batch_size,
            gamma=gamma,
            device=device,
        )

        result = env.get_result()
        reward = get_reward(result, double_dqn_agent.player)
        episode_rewards.append(reward)

        if episode_loss is not None:
            episode_losses.append(episode_loss)

        decay_epsilon(double_dqn_agent, epsilon_min, epsilon_decay)

        if episode % target_update_frequency == 0:
            update_target_network(q_network, target_network)

        if episode % checkpoint_frequency == 0:
            episode_checkpoint_path = os.path.join(
                checkpoint_dir,
                f"checkpoint_episode_{episode}.pth",
            )

            save_checkpoint(
                checkpoint_path=episode_checkpoint_path,
                episode=episode,
                q_network=q_network,
                target_network=target_network,
                optimizer=optimizer,
                replay_buffer=replay_buffer,
                epsilon=double_dqn_agent.epsilon,
                episode_rewards=episode_rewards,
                episode_losses=episode_losses,
            )

            save_checkpoint(
                checkpoint_path=latest_checkpoint_path,
                episode=episode,
                q_network=q_network,
                target_network=target_network,
                optimizer=optimizer,
                replay_buffer=replay_buffer,
                epsilon=double_dqn_agent.epsilon,
                episode_rewards=episode_rewards,
                episode_losses=episode_losses,
            )

            print(f"Checkpoint saved at episode {episode}")

        if episode % 100 == 0:
            recent_rewards = episode_rewards[-100:]
            average_reward = float(np.mean(recent_rewards))

            if episode_losses:
                recent_losses = episode_losses[-100:]
                average_loss = float(np.mean(recent_losses))
                print(
                    f"Episode {episode}/{num_episodes} | "
                    f"Avg Reward: {average_reward:.3f} | "
                    f"Avg Loss: {average_loss:.5f} | "
                    f"Epsilon: {double_dqn_agent.epsilon:.3f} | "
                    f"Replay Size: {len(replay_buffer)}"
                )
            else:
                print(
                    f"Episode {episode}/{num_episodes} | "
                    f"Avg Reward: {average_reward:.3f} | "
                    f"Epsilon: {double_dqn_agent.epsilon:.3f} | "
                    f"Replay Size: {len(replay_buffer)}"
                )

    final_model_path = os.path.join(checkpoint_dir, "double_dqn_model_final.pth")
    torch.save(q_network.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")

    return q_network, episode_rewards, episode_losses


def main():
    train_double_dqn(
        num_episodes=50000,
        checkpoint_frequency=500,
        resume=False,
    )


if __name__ == "__main__":
    main()
import os
import random

import numpy as np
import torch
import torch.optim as optim

from config import PLAYER_ONE, PLAYER_TWO
from env.connect4_env import Connect4Env
from agents.random_agent import RandomAgent
from agents.minimax_agent import MinimaxAgent
from agents.dqn_agent import DQNAgent
from models.q_network import QNetwork
from training.replay_buffer import ReplayBuffer
from training.training_common import (
    dqn_train_step,
    update_target_network,
    decay_epsilon,
    ensure_checkpoint_dir,
    save_checkpoint,
    get_reward,
    get_state,
)


def load_pretrained_model(model_path, q_network, device):
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    q_network.load_state_dict(state_dict)
    print(f"Loaded pretrained model from: {model_path}")


def select_opponent(prob_minimax=0.3):
    if random.random() < prob_minimax:
        return "minimax"
    return "random"


def play_training_episode(
    env,
    dqn_agent,
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

        if current_player == dqn_agent.player:
            state = get_state(env, dqn_agent.player)
            action = dqn_agent.select_action(env)

            if action is None:
                break

            env.drop_piece(action)
            result = env.get_result()

            if result is not None:
                reward = get_reward(result, dqn_agent.player)
                next_state = get_state(env, dqn_agent.player)

                replay_buffer.push(state, action, reward, next_state, [], True)

                loss = dqn_train_step(
                    q_network, target_network, replay_buffer,
                    optimizer, batch_size, gamma, device
                )
                if loss is not None:
                    episode_loss_values.append(loss)
                break

            opponent_action = opponent_agent.select_action(env)
            if opponent_action is None:
                break

            env.drop_piece(opponent_action)
            result = env.get_result()

            reward = get_reward(result, dqn_agent.player)
            next_state = get_state(env, dqn_agent.player)
            next_legal = env.get_legal_actions() if result is None else []

            replay_buffer.push(state, action, reward, next_state, next_legal, result is not None)

            loss = dqn_train_step(
                q_network, target_network, replay_buffer,
                optimizer, batch_size, gamma, device
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


def train_dqn_finetune(
    num_episodes=10000,
    batch_size=64,
    gamma=0.99,
    learning_rate=0.00005,
    replay_buffer_capacity=10000,
    target_update_frequency=100,
    epsilon_start=0.15,
    epsilon_min=0.05,
    epsilon_decay=0.9995,
    minimax_probability=0.3,
    checkpoint_dir="finetune_checkpoints",
    checkpoint_frequency=500,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ensure_checkpoint_dir(checkpoint_dir)

    env = Connect4Env()

    q_network = QNetwork().to(device)
    target_network = QNetwork().to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
    replay_buffer = ReplayBuffer(capacity=replay_buffer_capacity)

    # 🔥 LOAD IMITATION MODEL
    pretrained_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "training",
        "imitation_checkpoints",
        "dqn_imitation_mixed_model_final.pth",
    )
    load_pretrained_model(pretrained_path, q_network, device)

    update_target_network(q_network, target_network)

    dqn_agent = DQNAgent(
        player=PLAYER_ONE,
        q_network=q_network,
        epsilon=epsilon_start,
        device=device,
    )

    episode_rewards = []
    episode_losses = []

    print("\n=== Starting FINETUNING ===")

    for episode in range(1, num_episodes + 1):

        # Alternate player
        dqn_agent.player = PLAYER_ONE if episode % 2 == 1 else PLAYER_TWO

        opponent_type = select_opponent(minimax_probability)

        if opponent_type == "random":
            opponent = RandomAgent(player=-dqn_agent.player)
        else:
            opponent = MinimaxAgent(player=-dqn_agent.player, depth=2)

        loss = play_training_episode(
            env, dqn_agent, opponent,
            replay_buffer, q_network, target_network,
            optimizer, batch_size, gamma, device
        )

        result = env.get_result()
        reward = get_reward(result, dqn_agent.player)
        episode_rewards.append(reward)

        if loss is not None:
            episode_losses.append(loss)

        decay_epsilon(dqn_agent, epsilon_min, epsilon_decay)

        if episode % target_update_frequency == 0:
            update_target_network(q_network, target_network)

        if episode % checkpoint_frequency == 0:
            path = os.path.join(checkpoint_dir, f"checkpoint_{episode}.pth")
            save_checkpoint(
                path, episode,
                q_network, target_network,
                optimizer, replay_buffer,
                dqn_agent.epsilon,
                episode_rewards,
                episode_losses,
            )
            print(f"Checkpoint saved at episode {episode}")

        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(
                f"Episode {episode} | "
                f"Avg Reward: {avg_reward:.3f} | "
                f"Epsilon: {dqn_agent.epsilon:.3f}"
            )

    final_path = os.path.join(checkpoint_dir, "dqn_finetuned_model_final.pth")
    torch.save(q_network.state_dict(), final_path)

    print(f"\nFinal finetuned model saved to: {final_path}")


def main():
    train_dqn_finetune(
        num_episodes=10000,
    )


if __name__ == "__main__":
    main()
import os

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
    load_checkpoint,
    get_state,
)
from training.reward_functions import basic_reward, shaped_reward
from training.trainer_selfplay import play_selfplay_training_episode
from training.trainer_vs_minimax import (
    play_vs_minimax_training_episode,
    build_minimax_opponent,
)


def compute_reward(env, player, action, result, use_shaped_reward):
    if use_shaped_reward:
        return shaped_reward(env, player, action, result)
    return basic_reward(result, player)


def play_vs_random_training_episode(
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
    use_shaped_reward,
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
                reward = compute_reward(
                    env=env,
                    player=dqn_agent.player,
                    action=action,
                    result=result,
                    use_shaped_reward=use_shaped_reward,
                )
                next_state = get_state(env, dqn_agent.player)

                replay_buffer.push(state, action, reward, next_state, [], True)
                loss = dqn_train_step(
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
                next_state = get_state(env, dqn_agent.player)

                replay_buffer.push(state, action, 0.0, next_state, [], True)
                loss = dqn_train_step(
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

            reward = compute_reward(
                env=env,
                player=dqn_agent.player,
                action=action,
                result=result,
                use_shaped_reward=use_shaped_reward,
            )
            next_state = get_state(env, dqn_agent.player)
            next_legal_actions = env.get_legal_actions() if result is None else []
            done = result is not None

            replay_buffer.push(state, action, reward, next_state, next_legal_actions, done)
            loss = dqn_train_step(
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


def save_phase_model(checkpoint_dir, filename, q_network):
    model_path = os.path.join(checkpoint_dir, filename)
    torch.save(q_network.state_dict(), model_path)
    print(f"Phase model saved to {model_path}")


def run_phase(
    phase_name,
    start_episode,
    num_episodes,
    env,
    q_network,
    target_network,
    optimizer,
    replay_buffer,
    dqn_agent,
    batch_size,
    gamma,
    device,
    epsilon_min,
    epsilon_decay,
    target_update_frequency,
    checkpoint_dir,
    checkpoint_frequency,
    episode_rewards,
    episode_losses,
    use_shaped_reward,
):
    latest_checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pth")

    print(f"\n=== Starting phase: {phase_name} ({num_episodes} episodes) ===")

    for local_episode in range(1, num_episodes + 1):
        global_episode = start_episode + local_episode - 1

        if phase_name == "random":
            learning_player = PLAYER_ONE if global_episode % 2 == 1 else PLAYER_TWO
            dqn_agent.player = learning_player

            opponent_player = PLAYER_TWO if learning_player == PLAYER_ONE else PLAYER_ONE
            opponent_agent = RandomAgent(player=opponent_player)

            episode_loss = play_vs_random_training_episode(
                env=env,
                dqn_agent=dqn_agent,
                opponent_agent=opponent_agent,
                replay_buffer=replay_buffer,
                q_network=q_network,
                target_network=target_network,
                optimizer=optimizer,
                batch_size=batch_size,
                gamma=gamma,
                device=device,
                use_shaped_reward=use_shaped_reward,
            )

        elif phase_name == "selfplay":
            def selfplay_agent_factory(player):
                return DQNAgent(
                    player=player,
                    q_network=q_network,
                    epsilon=dqn_agent.epsilon,
                    device=device,
                )

            episode_loss = play_selfplay_training_episode(
                env=env,
                agent_factory=selfplay_agent_factory,
                replay_buffer=replay_buffer,
                q_network=q_network,
                target_network=target_network,
                optimizer=optimizer,
                train_step_fn=dqn_train_step,
                batch_size=batch_size,
                gamma=gamma,
                device=device,
            )

        elif phase_name == "minimax":
            learning_player = PLAYER_ONE if global_episode % 2 == 1 else PLAYER_TWO
            dqn_agent.player = learning_player

            opponent_agent = build_minimax_opponent(
                minimax_class=MinimaxAgent,
                learning_player=learning_player,
                depth=2,
            )

            episode_loss = play_vs_minimax_training_episode(
                env=env,
                learning_agent=dqn_agent,
                opponent_agent=opponent_agent,
                replay_buffer=replay_buffer,
                q_network=q_network,
                target_network=target_network,
                optimizer=optimizer,
                train_step_fn=dqn_train_step,
                batch_size=batch_size,
                gamma=gamma,
                device=device,
            )

        else:
            raise ValueError(f"Unknown phase: {phase_name}")

        result = env.get_result()

        if phase_name == "selfplay":
            if result == 0 or result is None:
                reward = 0.0
            else:
                reward = 1.0
        else:
            reward = basic_reward(result, dqn_agent.player)

        episode_rewards.append(reward)

        if episode_loss is not None:
            episode_losses.append(episode_loss)

        decay_epsilon(dqn_agent, epsilon_min, epsilon_decay)

        if global_episode % target_update_frequency == 0:
            update_target_network(q_network, target_network)

        if global_episode % checkpoint_frequency == 0:
            episode_checkpoint_path = os.path.join(
                checkpoint_dir,
                f"checkpoint_episode_{global_episode}.pth",
            )

            save_checkpoint(
                checkpoint_path=episode_checkpoint_path,
                episode=global_episode,
                q_network=q_network,
                target_network=target_network,
                optimizer=optimizer,
                replay_buffer=replay_buffer,
                epsilon=dqn_agent.epsilon,
                episode_rewards=episode_rewards,
                episode_losses=episode_losses,
            )

            save_checkpoint(
                checkpoint_path=latest_checkpoint_path,
                episode=global_episode,
                q_network=q_network,
                target_network=target_network,
                optimizer=optimizer,
                replay_buffer=replay_buffer,
                epsilon=dqn_agent.epsilon,
                episode_rewards=episode_rewards,
                episode_losses=episode_losses,
            )

            print(f"Checkpoint saved at episode {global_episode}")

        if global_episode % 100 == 0:
            recent_rewards = episode_rewards[-100:]
            average_reward = float(np.mean(recent_rewards))

            if episode_losses:
                recent_losses = episode_losses[-100:]
                average_loss = float(np.mean(recent_losses))
                print(
                    f"[{phase_name.upper()}] "
                    f"Episode {global_episode} | "
                    f"Avg Reward: {average_reward:.3f} | "
                    f"Avg Loss: {average_loss:.5f} | "
                    f"Epsilon: {dqn_agent.epsilon:.3f} | "
                    f"Replay Size: {len(replay_buffer)}"
                )
            else:
                print(
                    f"[{phase_name.upper()}] "
                    f"Episode {global_episode} | "
                    f"Avg Reward: {average_reward:.3f} | "
                    f"Epsilon: {dqn_agent.epsilon:.3f} | "
                    f"Replay Size: {len(replay_buffer)}"
                )

    return start_episode + num_episodes


def train_dqn(
    random_phase_episodes=10000,
    selfplay_phase_episodes=20000,
    minimax_phase_episodes=20000,
    batch_size=64,
    gamma=0.99,
    learning_rate=0.0001,
    replay_buffer_capacity=10000,
    target_update_frequency=100,
    epsilon_start=1.0,
    epsilon_min=0.10,
    epsilon_decay=0.9999,
    checkpoint_dir="checkpoints_curriculum",
    checkpoint_frequency=500,
    resume=False,
    checkpoint_path=None,
    use_shaped_reward=True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ensure_checkpoint_dir(checkpoint_dir)

    env = Connect4Env()

    q_network = QNetwork().to(device)
    target_network = QNetwork().to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
    replay_buffer = ReplayBuffer(capacity=replay_buffer_capacity)

    update_target_network(q_network, target_network)

    dqn_agent = DQNAgent(
        player=PLAYER_ONE,
        q_network=q_network,
        epsilon=epsilon_start,
        device=device,
    )

    episode_rewards = []
    episode_losses = []
    start_episode = 1

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
            dqn_agent.epsilon = loaded_epsilon
            print(f"Resumed training from episode {start_episode}")
        else:
            print("No checkpoint found. Starting training from scratch.")

    phase_plan = [
        ("random", random_phase_episodes),
        ("selfplay", selfplay_phase_episodes),
        ("minimax", minimax_phase_episodes),
    ]

    phase_epsilon_starts = {
        "random": 1.0,
        "selfplay": 0.30,
        "minimax": 0.20,
    }

    current_start_episode = start_episode

    reward_mode = "shaped" if use_shaped_reward else "basic"
    print(f"\nUsing {reward_mode} reward for random-phase training.")

    for phase_name, num_phase_episodes in phase_plan:
        if num_phase_episodes <= 0:
            print(f"Skipping phase: {phase_name}")
            continue

        dqn_agent.epsilon = phase_epsilon_starts[phase_name]
        print(f"\nReset epsilon for {phase_name} phase to {dqn_agent.epsilon:.3f}")

        current_start_episode = run_phase(
            phase_name=phase_name,
            start_episode=current_start_episode,
            num_episodes=num_phase_episodes,
            env=env,
            q_network=q_network,
            target_network=target_network,
            optimizer=optimizer,
            replay_buffer=replay_buffer,
            dqn_agent=dqn_agent,
            batch_size=batch_size,
            gamma=gamma,
            device=device,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            target_update_frequency=target_update_frequency,
            checkpoint_dir=checkpoint_dir,
            checkpoint_frequency=checkpoint_frequency,
            episode_rewards=episode_rewards,
            episode_losses=episode_losses,
            use_shaped_reward=use_shaped_reward,
        )

        if phase_name == "random":
            save_phase_model(checkpoint_dir, "dqn_model_after_random.pth", q_network)
        elif phase_name == "selfplay":
            save_phase_model(checkpoint_dir, "dqn_model_after_selfplay.pth", q_network)
        elif phase_name == "minimax":
            save_phase_model(checkpoint_dir, "dqn_model_after_minimax.pth", q_network)

    final_model_path = os.path.join(checkpoint_dir, "dqn_model_final.pth")
    torch.save(q_network.state_dict(), final_model_path)
    print(f"\nFinal model saved to {final_model_path}")

    return q_network, episode_rewards, episode_losses


def main():
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "full"

    if mode == "random":
        print("\nRunning RANDOM-ONLY training")
        train_dqn(
            random_phase_episodes=10000,
            selfplay_phase_episodes=0,
            minimax_phase_episodes=0,
            checkpoint_dir="checkpoints_curriculum_random_only",
            checkpoint_frequency=500,
            resume=False,
            use_shaped_reward=True,
        )

    elif mode == "selfplay":
        print("\nRunning RANDOM + SELF-PLAY training")
        train_dqn(
            random_phase_episodes=10000,
            selfplay_phase_episodes=20000,
            minimax_phase_episodes=0,
            checkpoint_dir="checkpoints_curriculum_random_selfplay",
            checkpoint_frequency=500,
            resume=False,
            use_shaped_reward=True,
        )

    elif mode == "full":
        print("\nRunning FULL curriculum training")
        train_dqn(
            random_phase_episodes=10000,
            selfplay_phase_episodes=20000,
            minimax_phase_episodes=20000,
            checkpoint_dir="checkpoints_curriculum_full",
            checkpoint_frequency=500,
            resume=False,
            use_shaped_reward=True,
        )

    else:
        print("Invalid mode. Use: random | selfplay | full")


if __name__ == "__main__":
    main()
import numpy as np

from config import PLAYER_ONE, PLAYER_TWO
from training.training_common import get_state, get_reward


def play_vs_minimax_training_episode(
    env,
    learning_agent,
    opponent_agent,
    replay_buffer,
    q_network,
    target_network,
    optimizer,
    train_step_fn,
    batch_size,
    gamma,
    device,
):
    env.reset()
    episode_loss_values = []

    while not env.game_over:
        current_player = env.get_current_player()

        if current_player == learning_agent.player:
            state = get_state(env, learning_agent.player)
            action = learning_agent.select_action(env)

            if action is None:
                break

            env.drop_piece(action)
            result = env.get_result()

            if result is not None:
                reward = get_reward(result, learning_agent.player)
                next_state = get_state(env, learning_agent.player)

                replay_buffer.push(state, action, reward, next_state, [], True)
                loss = train_step_fn(
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
                next_state = get_state(env, learning_agent.player)
                replay_buffer.push(state, action, 0.0, next_state, [], True)
                loss = train_step_fn(
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

            reward = get_reward(result, learning_agent.player)
            next_state = get_state(env, learning_agent.player)
            next_legal_actions = env.get_legal_actions() if result is None else []
            done = result is not None

            replay_buffer.push(state, action, reward, next_state, next_legal_actions, done)
            loss = train_step_fn(
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


def build_minimax_opponent(minimax_class, learning_player, depth):
    opponent_player = PLAYER_TWO if learning_player == PLAYER_ONE else PLAYER_ONE
    return minimax_class(player=opponent_player, depth=depth)
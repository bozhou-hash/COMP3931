import numpy as np

from training.training_common import get_state, get_reward


def play_selfplay_training_episode(
    env,
    agent_factory,
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

    agent_player_one = agent_factory(player=1)
    agent_player_two = agent_factory(player=-1)

    pending_transition = {
        1: None,
        -1: None,
    }

    while not env.game_over:
        current_player = env.get_current_player()

        if current_player == 1:
            acting_agent = agent_player_one
            opponent_player = -1
        else:
            acting_agent = agent_player_two
            opponent_player = 1

        state = get_state(env, current_player)
        action = acting_agent.select_action(env)

        if action is None:
            break

        env.drop_piece(action)
        result = env.get_result()

        # Finalize the acting player's transition
        if result is not None:
            reward = get_reward(result, current_player)
            next_state = get_state(env, current_player)
            next_legal_actions = []
            done = True

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

            # Finalize opponent's previous pending move, if there was one
            if pending_transition[opponent_player] is not None:
                prev_state, prev_action = pending_transition[opponent_player]
                opponent_reward = get_reward(result, opponent_player)
                opponent_next_state = get_state(env, opponent_player)

                replay_buffer.push(
                    prev_state,
                    prev_action,
                    opponent_reward,
                    opponent_next_state,
                    [],
                    True,
                )
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

        # If game continues, finalize the opponent's previous pending move
        if pending_transition[opponent_player] is not None:
            prev_state, prev_action = pending_transition[opponent_player]
            opponent_next_state = get_state(env, opponent_player)
            opponent_next_legal_actions = env.get_legal_actions()

            replay_buffer.push(
                prev_state,
                prev_action,
                0.0,
                opponent_next_state,
                opponent_next_legal_actions,
                False,
            )
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

            pending_transition[opponent_player] = None

        # Store current move as pending until opponent responds
        pending_transition[current_player] = (state, action)

    if not episode_loss_values:
        return None

    return float(np.mean(episode_loss_values))
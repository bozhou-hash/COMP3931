import os
import random

import numpy as np

from config import PLAYER_ONE, PLAYER_TWO
from env.connect4_env import Connect4Env
from agents.minimax_agent import MinimaxAgent
from training.training_common import get_state


def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_dataset_path(filename):
    dataset_dir = os.path.join(get_project_root(), "datasets")
    os.makedirs(dataset_dir, exist_ok=True)
    return os.path.join(dataset_dir, filename)


def clone_env(env):
    cloned = Connect4Env()
    cloned.board = env.get_board().copy()
    cloned.current_player = env.get_current_player()
    cloned.game_over = env.game_over
    cloned.winner = env.winner
    cloned.starting_player = env.starting_player
    return cloned


def play_random_moves(env, num_random_moves):
    for _ in range(num_random_moves):
        if env.game_over:
            break
        legal_actions = env.get_legal_actions()
        if not legal_actions:
            break
        action = random.choice(legal_actions)
        env.drop_piece(action)


def find_immediate_winning_moves(env, player):
    winning_moves = []

    for action in env.get_legal_actions():
        temp_env = clone_env(env)
        if temp_env.get_current_player() != player:
            temp_env.current_player = player

        temp_env.drop_piece(action)
        if temp_env.get_result() == player:
            winning_moves.append(action)

    return winning_moves


def is_tactical_position(env):
    current_player = env.get_current_player()
    opponent = -current_player

    current_wins = find_immediate_winning_moves(env, current_player)
    opponent_wins = find_immediate_winning_moves(env, opponent)

    return len(current_wins) > 0 or len(opponent_wins) > 0


def generate_minimax_tactical_dataset(
    num_samples=50000,
    minimax_depth=4,
    max_random_moves=12,
    save_filename="minimax_tactical_dataset.npz",
):
    env = Connect4Env()

    states = []
    actions = []

    minimax_agent_p1 = MinimaxAgent(player=PLAYER_ONE, depth=minimax_depth)
    minimax_agent_p2 = MinimaxAgent(player=PLAYER_TWO, depth=minimax_depth)

    print(
        f"Generating tactical Minimax dataset with {num_samples} samples "
        f"(depth={minimax_depth}, max_random_moves={max_random_moves})..."
    )

    attempts = 0
    while len(states) < num_samples:
        attempts += 1
        env.reset()

        num_random_moves = random.randint(0, max_random_moves)
        play_random_moves(env, num_random_moves)

        if env.game_over:
            continue

        if not is_tactical_position(env):
            continue

        current_player = env.get_current_player()
        agent = minimax_agent_p1 if current_player == PLAYER_ONE else minimax_agent_p2

        state = get_state(env, current_player)
        action = agent.select_action(env)

        if action is None:
            continue

        states.append(state)
        actions.append(action)

        if len(states) % 5000 == 0:
            print(f"Collected {len(states)}/{num_samples} samples")

    states = np.array(states, dtype=np.float32)
    actions = np.array(actions, dtype=np.int64)

    dataset_path = get_dataset_path(save_filename)
    np.savez(dataset_path, states=states, actions=actions)

    print(f"\nDataset saved to: {dataset_path}")
    print(f"Total samples: {len(states)}")
    print(f"Generation attempts: {attempts}")


def main():
    generate_minimax_tactical_dataset(
        num_samples=100000,
        minimax_depth=6,
        max_random_moves=12,
        save_filename="minimax_tactical_dataset_depth6.npz",
    )


if __name__ == "__main__":
    main()
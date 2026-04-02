import os
import numpy as np
import torch

from env.connect4_env import Connect4Env
from agents.minimax_agent import MinimaxAgent
from training.training_common import get_state
from config import PLAYER_ONE, PLAYER_TWO


def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_dataset_path(filename):
    dataset_dir = os.path.join(get_project_root(), "datasets")
    os.makedirs(dataset_dir, exist_ok=True)
    return os.path.join(dataset_dir, filename)


def generate_minimax_dataset(
    num_games=5000,
    minimax_depth=3,
    save_filename="minimax_dataset.npz",
):
    env = Connect4Env()

    states = []
    actions = []

    print(f"Generating dataset with {num_games} games (depth={minimax_depth})...")

    for game_idx in range(1, num_games + 1):
        env.reset()

        minimax_agent_p1 = MinimaxAgent(player=PLAYER_ONE, depth=minimax_depth)
        minimax_agent_p2 = MinimaxAgent(player=PLAYER_TWO, depth=minimax_depth)

        while not env.game_over:
            current_player = env.get_current_player()

            if current_player == PLAYER_ONE:
                agent = minimax_agent_p1
            else:
                agent = minimax_agent_p2

            state = get_state(env, current_player)
            action = agent.select_action(env)

            if action is None:
                break

            # Store (state, action)
            states.append(state)
            actions.append(action)

            env.drop_piece(action)

        if game_idx % 100 == 0:
            print(f"Generated {game_idx}/{num_games} games")

    states = np.array(states, dtype=np.float32)
    actions = np.array(actions, dtype=np.int64)

    dataset_path = get_dataset_path(save_filename)
    np.savez(dataset_path, states=states, actions=actions)

    print(f"\nDataset saved to: {dataset_path}")
    print(f"Total samples: {len(states)}")


def main():
    generate_minimax_dataset(
        num_games=5000,       # You can increase later (e.g., 10000+)
        minimax_depth=3,      # Start with 3 for speed
        save_filename="minimax_dataset.npz",
    )


if __name__ == "__main__":
    main()
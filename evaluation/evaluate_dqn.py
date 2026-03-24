import os

import torch

from config import PLAYER_ONE, PLAYER_TWO
from env.connect4_env import Connect4Env
from agents.random_agent import RandomAgent
from agents.minimax_agent import MinimaxAgent
from agents.dqn_agent import DQNAgent
from models.q_network import QNetwork


def get_model_path(filename):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, "training", "checkpoints", filename)


def load_dqn_agent(model_path, player, device):
    q_network = QNetwork().to(device)

    if model_path.endswith("dqn_model_final.pth"):
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        q_network.load_state_dict(state_dict)
    else:
        checkpoint = torch.load(
            model_path,
            map_location=device,
            weights_only=False,
        )
        q_network.load_state_dict(checkpoint["q_network_state_dict"])

    q_network.eval()

    return DQNAgent(
        player=player,
        q_network=q_network,
        epsilon=0.0,
        device=device,
    )


def play_game(env, agent_one, agent_two):
    env.reset()

    while not env.game_over:
        current_player = env.get_current_player()

        if current_player == agent_one.player:
            action = agent_one.select_action(env)
        else:
            action = agent_two.select_action(env)

        if action is None:
            break

        env.drop_piece(action)

    return env.get_result()


def evaluate_matchup(env, agent_one_factory, agent_two_factory, num_games):
    agent_one_wins = 0
    agent_two_wins = 0
    draws = 0

    for game_index in range(num_games):
        if game_index % 2 == 0:
            agent_one = agent_one_factory(PLAYER_ONE)
            agent_two = agent_two_factory(PLAYER_TWO)
        else:
            agent_one = agent_one_factory(PLAYER_TWO)
            agent_two = agent_two_factory(PLAYER_ONE)

        result = play_game(env, agent_one, agent_two)

        if result == agent_one.player:
            agent_one_wins += 1
        elif result == agent_two.player:
            agent_two_wins += 1
        else:
            draws += 1

    return {
        "num_games": num_games,
        "agent_one_wins": agent_one_wins,
        "agent_two_wins": agent_two_wins,
        "draws": draws,
        "agent_one_win_rate": agent_one_wins / num_games,
        "agent_two_win_rate": agent_two_wins / num_games,
        "draw_rate": draws / num_games,
    }


def print_results(title, agent_one_name, agent_two_name, results):
    print(f"\n=== {title} ===")
    print(f"Games played: {results['num_games']}")
    print(f"{agent_one_name} wins: {results['agent_one_wins']}")
    print(f"{agent_two_name} wins: {results['agent_two_wins']}")
    print(f"Draws: {results['draws']}")
    print(f"{agent_one_name} win rate: {results['agent_one_win_rate']:.2%}")
    print(f"{agent_two_name} win rate: {results['agent_two_win_rate']:.2%}")
    print(f"Draw rate: {results['draw_rate']:.2%}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_filename = "dqn_model_final.pth"
    model_path = get_model_path(model_filename)

    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return

    print(f"Evaluating model: {model_path}")

    env = Connect4Env()

    dqn_factory = lambda player: load_dqn_agent(model_path, player, device)
    random_factory = lambda player: RandomAgent(player=player)
    minimax_factory = lambda player: MinimaxAgent(player=player, depth=4)

    dqn_vs_random_results = evaluate_matchup(
        env=env,
        agent_one_factory=dqn_factory,
        agent_two_factory=random_factory,
        num_games=100,
    )
    print_results("DQN vs Random", "DQN", "Random", dqn_vs_random_results)

    dqn_vs_minimax_results = evaluate_matchup(
        env=env,
        agent_one_factory=dqn_factory,
        agent_two_factory=minimax_factory,
        num_games=100,
    )
    print_results("DQN vs Minimax", "DQN", "Minimax", dqn_vs_minimax_results)


if __name__ == "__main__":
    main()
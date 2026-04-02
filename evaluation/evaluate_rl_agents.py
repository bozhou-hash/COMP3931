import json
import os

import torch

from config import PLAYER_ONE, PLAYER_TWO
from env.connect4_env import Connect4Env
from agents.random_agent import RandomAgent
from agents.minimax_agent import MinimaxAgent
from agents.dqn_agent import DQNAgent
from agents.double_dqn_agent import DoubleDQNAgent
from models.q_network import QNetwork
from agents.hybrid_agent import HybridAgent


def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_results_dir(results_subdir="default"):
    results_dir = os.path.join(get_project_root(), "results", results_subdir)
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def get_model_path(folder_name, filename):
    base_dir = get_project_root()
    return os.path.join(base_dir, "training", folder_name, filename)


def load_q_network(model_path, device):
    q_network = QNetwork().to(device)

    loaded_object = torch.load(model_path, map_location=device, weights_only=False)

    # Case 1: plain state_dict
    if isinstance(loaded_object, dict) and "q_network_state_dict" not in loaded_object:
        q_network.load_state_dict(loaded_object)

    # Case 2: checkpoint dict
    elif isinstance(loaded_object, dict) and "q_network_state_dict" in loaded_object:
        q_network.load_state_dict(loaded_object["q_network_state_dict"])

    else:
        raise ValueError(f"Unsupported model file format: {model_path}")

    q_network.eval()
    return q_network


def create_dqn_factory(q_network, device):
    return lambda player: DQNAgent(
        player=player,
        q_network=q_network,
        epsilon=0.0,
        device=device,
    )


def create_double_dqn_factory(q_network, device):
    return lambda player: DoubleDQNAgent(
        player=player,
        q_network=q_network,
        epsilon=0.0,
        device=device,
    )


def create_hybrid_factory(q_network, device, depth=4, neural_weight=0.7, heuristic_weight=0.3):
    return lambda player: HybridAgent(
        player=player,
        q_network=q_network,
        device=device,
        depth=depth,
        neural_weight=neural_weight,
        heuristic_weight=heuristic_weight,
    )


def evaluate_matchup(env, agent_a_factory, agent_b_factory, num_games):
    stats = {
        "num_games": num_games,
        "agent_a_wins": 0,
        "agent_b_wins": 0,
        "draws": 0,
        "agent_a_as_first_wins": 0,
        "agent_a_as_second_wins": 0,
        "agent_b_as_first_wins": 0,
        "agent_b_as_second_wins": 0,
        "first_player_wins": 0,
        "second_player_wins": 0,
    }

    for _ in range(num_games):
        env.reset()
        starting_player = env.get_current_player()

        agent_a = agent_a_factory(PLAYER_ONE)
        agent_b = agent_b_factory(PLAYER_TWO)

        if starting_player == PLAYER_ONE:
            first_agent = "agent_a"
        else:
            first_agent = "agent_b"

        while not env.game_over:
            current_player = env.get_current_player()

            if current_player == agent_a.player:
                action = agent_a.select_action(env)
            else:
                action = agent_b.select_action(env)

            if action is None:
                break

            env.drop_piece(action)

        result = env.get_result()

        if result == agent_a.player:
            stats["agent_a_wins"] += 1
            if first_agent == "agent_a":
                stats["agent_a_as_first_wins"] += 1
                stats["first_player_wins"] += 1
            else:
                stats["agent_a_as_second_wins"] += 1
                stats["second_player_wins"] += 1

        elif result == agent_b.player:
            stats["agent_b_wins"] += 1
            if first_agent == "agent_b":
                stats["agent_b_as_first_wins"] += 1
                stats["first_player_wins"] += 1
            else:
                stats["agent_b_as_second_wins"] += 1
                stats["second_player_wins"] += 1

        else:
            stats["draws"] += 1

    stats["agent_a_win_rate"] = stats["agent_a_wins"] / num_games
    stats["agent_b_win_rate"] = stats["agent_b_wins"] / num_games
    stats["draw_rate"] = stats["draws"] / num_games
    stats["first_player_win_rate"] = stats["first_player_wins"] / num_games
    stats["second_player_win_rate"] = stats["second_player_wins"] / num_games

    return stats


def save_results(filename, results, results_subdir="default"):
    results_path = os.path.join(get_results_dir(results_subdir), filename)

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to: {results_path}")


def print_results(title, agent_a_name, agent_b_name, results):
    print(f"\n=== {title} ===")
    print(f"Games played: {results['num_games']}")
    print(f"{agent_a_name} wins: {results['agent_a_wins']}")
    print(f"{agent_b_name} wins: {results['agent_b_wins']}")
    print(f"Draws: {results['draws']}")
    print(f"{agent_a_name} win rate: {results['agent_a_win_rate']:.2%}")
    print(f"{agent_b_name} win rate: {results['agent_b_win_rate']:.2%}")
    print(f"Draw rate: {results['draw_rate']:.2%}")
    print(f"{agent_a_name} wins as first player: {results['agent_a_as_first_wins']}")
    print(f"{agent_a_name} wins as second player: {results['agent_a_as_second_wins']}")
    print(f"{agent_b_name} wins as first player: {results['agent_b_as_first_wins']}")
    print(f"{agent_b_name} wins as second player: {results['agent_b_as_second_wins']}")
    print(f"First player overall win rate: {results['first_player_win_rate']:.2%}")
    print(f"Second player overall win rate: {results['second_player_win_rate']:.2%}")


def run_matchup(
    env,
    title,
    file_name,
    agent_a_name,
    agent_b_name,
    agent_a_factory,
    agent_b_factory,
    num_games,
    results_subdir="default",
):
    results = evaluate_matchup(
        env=env,
        agent_a_factory=agent_a_factory,
        agent_b_factory=agent_b_factory,
        num_games=num_games,
    )
    print_results(title, agent_a_name, agent_b_name, results)
    save_results(file_name, results, results_subdir=results_subdir)


def main():
    import sys

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_games = 500

    mode = sys.argv[1] if len(sys.argv) > 1 else "baseline"

    if mode == "baseline":
        dqn_model_path = get_model_path("checkpoints", "dqn_model_final.pth")
        double_dqn_model_path = get_model_path("double_dqn_checkpoints", "double_dqn_model_final.pth")
        dqn_name = "DQN"
        results_subdir = "baseline_eval"

    elif mode == "random":
        dqn_model_path = get_model_path("checkpoints_curriculum_random_only", "dqn_model_final.pth")
        double_dqn_model_path = get_model_path("double_dqn_checkpoints", "double_dqn_model_final.pth")
        dqn_name = "DQN"
        results_subdir = "curriculum_random_eval"

    elif mode == "selfplay":
        dqn_model_path = get_model_path("checkpoints_curriculum_random_selfplay", "dqn_model_final.pth")
        double_dqn_model_path = get_model_path("double_dqn_checkpoints", "double_dqn_model_final.pth")
        dqn_name = "DQN"
        results_subdir = "curriculum_selfplay_eval"

    elif mode == "full":
        dqn_model_path = get_model_path("checkpoints_curriculum_full", "dqn_model_final.pth")
        double_dqn_model_path = get_model_path("double_dqn_checkpoints", "double_dqn_model_final.pth")
        dqn_name = "DQN"
        results_subdir = "curriculum_full_eval"

    elif mode == "imitation":
        dqn_model_path = get_model_path("imitation_checkpoints", "dqn_imitation_model_final.pth")
        double_dqn_model_path = get_model_path("double_dqn_checkpoints", "double_dqn_model_final.pth")
        dqn_name = "Imitation DQN"
        results_subdir = "imitation_eval"

    elif mode == "imitation_mixed":
        dqn_model_path = get_model_path("imitation_checkpoints", "dqn_imitation_mixed_model_final.pth")
        double_dqn_model_path = get_model_path("double_dqn_checkpoints", "double_dqn_model_final.pth")
        dqn_name = "Imitation Mixed DQN"
        results_subdir = "imitation_mixed_eval"

    elif mode == "finetune":
        dqn_model_path = get_model_path("finetune_checkpoints", "dqn_finetuned_model_final.pth")
        double_dqn_model_path = get_model_path("double_dqn_checkpoints", "double_dqn_model_final.pth")
        dqn_name = "Finetuned DQN"
        results_subdir = "finetune_eval"

    elif mode == "imitation_tactical":
        dqn_model_path = get_model_path("imitation_checkpoints", "dqn_imitation_tactical_model_final.pth")
        double_dqn_model_path = get_model_path("double_dqn_checkpoints", "double_dqn_model_final.pth")
        dqn_name = "Imitation Tactical DQN"
        results_subdir = "imitation_tactical_eval"

    elif mode == "imitation_tactical_depth5":
        dqn_model_path = get_model_path("imitation_checkpoints", "dqn_imitation_tactical_depth5_model_final.pth")
        double_dqn_model_path = get_model_path("double_dqn_checkpoints", "double_dqn_model_final.pth")
        dqn_name = "Imitation Tactical Depth5 DQN"
        results_subdir = "imitation_tactical_depth5_eval"

    elif mode == "imitation_tactical_depth6":
        dqn_model_path = get_model_path("imitation_checkpoints", "dqn_imitation_tactical_depth6_model_final.pth")
        double_dqn_model_path = get_model_path("double_dqn_checkpoints", "double_dqn_model_final.pth")
        dqn_name = "Imitation Tactical Depth6 DQN"
        results_subdir = "imitation_tactical_depth6_eval"

    elif mode == "hybrid":
        dqn_model_path = get_model_path("imitation_checkpoints", "dqn_imitation_tactical_depth6_model_final.pth")
        double_dqn_model_path = get_model_path("double_dqn_checkpoints", "double_dqn_model_final.pth")
        dqn_name = "Hybrid DQN"
        results_subdir = "hybrid_eval_depth6"

    else:
        print("Invalid mode. Use: baseline | random | selfplay | full | imitation | imitation_mixed | finetune | imitation_tactical | imitation_tactical_depth5 | imitation_tactical_depth6 | hybrid")
        return

    if not os.path.exists(dqn_model_path):
        print(f"DQN model file not found: {dqn_model_path}")
        return

    if not os.path.exists(double_dqn_model_path):
        print(f"Double DQN model file not found: {double_dqn_model_path}")
        return

    print(f"Evaluating {dqn_name} model: {dqn_model_path}")
    print(f"Evaluating Double DQN model: {double_dqn_model_path}")
    print(f"Saving results under: results/{results_subdir}")

    dqn_q_network = load_q_network(dqn_model_path, device)
    double_dqn_q_network = load_q_network(double_dqn_model_path, device)

    env = Connect4Env()

    if mode == "hybrid":
        dqn_factory = create_hybrid_factory(
            dqn_q_network,
            device,
            depth=4,
            neural_weight=0.7,
            heuristic_weight=0.3,
        )
    else:
        dqn_factory = create_dqn_factory(dqn_q_network, device)

    double_dqn_factory = create_double_dqn_factory(double_dqn_q_network, device)

    random_factory = lambda player: RandomAgent(player=player)
    minimax_factory = lambda player: MinimaxAgent(player=player, depth=6)

    run_matchup(
        env=env,
        title=f"{dqn_name} vs Random",
        file_name="dqn_vs_random.json",
        agent_a_name=dqn_name,
        agent_b_name="Random",
        agent_a_factory=dqn_factory,
        agent_b_factory=random_factory,
        num_games=num_games,
        results_subdir=results_subdir,
    )

    run_matchup(
        env=env,
        title="Double DQN vs Random",
        file_name="double_dqn_vs_random.json",
        agent_a_name="Double DQN",
        agent_b_name="Random",
        agent_a_factory=double_dqn_factory,
        agent_b_factory=random_factory,
        num_games=num_games,
        results_subdir=results_subdir,
    )

    run_matchup(
        env=env,
        title=f"{dqn_name} vs Minimax",
        file_name="dqn_vs_minimax.json",
        agent_a_name=dqn_name,
        agent_b_name="Minimax",
        agent_a_factory=dqn_factory,
        agent_b_factory=minimax_factory,
        num_games=num_games,
        results_subdir=results_subdir,
    )

    run_matchup(
        env=env,
        title="Double DQN vs Minimax",
        file_name="double_dqn_vs_minimax.json",
        agent_a_name="Double DQN",
        agent_b_name="Minimax",
        agent_a_factory=double_dqn_factory,
        agent_b_factory=minimax_factory,
        num_games=num_games,
        results_subdir=results_subdir,
    )

    run_matchup(
        env=env,
        title=f"Double DQN vs {dqn_name}",
        file_name="double_dqn_vs_dqn.json",
        agent_a_name="Double DQN",
        agent_b_name=dqn_name,
        agent_a_factory=double_dqn_factory,
        agent_b_factory=dqn_factory,
        num_games=num_games,
        results_subdir=results_subdir,
    )


if __name__ == "__main__":
    main()
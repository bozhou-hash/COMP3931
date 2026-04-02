import os

import torch

from env.connect4_env import Connect4Env
from ui.game_ui import GameUI
from agents.random_agent import RandomAgent
from agents.minimax_agent import MinimaxAgent
from agents.dqn_agent import DQNAgent
from agents.double_dqn_agent import DoubleDQNAgent
from models.q_network import QNetwork
from config import PLAYER_ONE, PLAYER_TWO
from agents.tactical_wrapper_agent import TacticalWrapperAgent
from agents.hybrid_agent import HybridAgent


def get_project_root():
    return os.path.dirname(os.path.abspath(__file__))


def get_model_path(folder_name, filename):
    return os.path.join(get_project_root(), "training", folder_name, filename)


def load_q_network(model_path, device):
    q_network = QNetwork().to(device)

    loaded_object = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(loaded_object, dict) and "q_network_state_dict" in loaded_object:
        q_network.load_state_dict(loaded_object["q_network_state_dict"])
    elif isinstance(loaded_object, dict):
        q_network.load_state_dict(loaded_object)
    else:
        raise ValueError(f"Unsupported model format: {model_path}")

    q_network.eval()
    return q_network


def load_dqn_agent(model_folder, model_filename, player, device):
    model_path = get_model_path(model_folder, model_filename)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"DQN model file not found: {model_path}")

    q_network = load_q_network(model_path, device)

    return DQNAgent(
        player=player,
        q_network=q_network,
        epsilon=0.0,
        device=device,
    )


def load_double_dqn_agent(model_folder, model_filename, player, device):
    model_path = get_model_path(model_folder, model_filename)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Double DQN model file not found: {model_path}")

    q_network = load_q_network(model_path, device)

    return DoubleDQNAgent(
        player=player,
        q_network=q_network,
        epsilon=0.0,
        device=device,
    )

def load_hybrid_agent(model_folder, model_filename, player, device, depth=4):
    model_path = get_model_path(model_folder, model_filename)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Hybrid model file not found: {model_path}")

    q_network = load_q_network(model_path, device)

    return HybridAgent(
        player=player,
        q_network=q_network,
        device=device,
        depth=depth,
        neural_weight=0.7,
        heuristic_weight=0.3,
    )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = Connect4Env()
    ui = GameUI(env)

    # 1. Human vs Human
    # ui.run_human_vs_human()

    # 2. Human vs Random
    # random_agent = RandomAgent(player=PLAYER_TWO)
    # ui.run_human_vs_agent(agent=random_agent, human_player=PLAYER_ONE)

    # 3. Human vs Minimax depth 4
    # minimax_agent = MinimaxAgent(player=PLAYER_TWO, depth=4)
    # ui.run_human_vs_agent(agent=minimax_agent, human_player=PLAYER_ONE)

    # 4. Human vs baseline DQN
    # dqn_agent = load_dqn_agent(
    #     model_folder="checkpoints",
    #     model_filename="dqn_model_final.pth",
    #     player=PLAYER_TWO,
    #     device=device,
    # )
    # ui.run_human_vs_agent(agent=dqn_agent, human_player=PLAYER_ONE)

    # 5. Human vs curriculum random-only DQN
    # dqn_agent = load_dqn_agent(
    #     model_folder="checkpoints_curriculum_random_only",
    #     model_filename="dqn_model_final.pth",
    #     player=PLAYER_TWO,
    #     device=device,
    # )
    # ui.run_human_vs_agent(agent=dqn_agent, human_player=PLAYER_ONE)

    # 6. Human vs Double DQN
    # double_dqn_agent = load_double_dqn_agent(
    #     model_folder="double_dqn_checkpoints",
    #     model_filename="double_dqn_model_final.pth",
    #     player=PLAYER_TWO,
    #     device=device,
    # )
    # ui.run_human_vs_agent(agent=double_dqn_agent, human_player=PLAYER_ONE)

    # 7. Human vs Imitation Mixed DQN
    # dqn_agent = load_dqn_agent(
    #     model_folder="imitation_checkpoints",
    #     model_filename="dqn_imitation_mixed_model_final.pth",
    #     player=PLAYER_TWO,
    #     device=device,
    # )
    # ui.run_human_vs_agent(agent=dqn_agent, human_player=PLAYER_ONE)

    # 8. Human vs Imitation Tactical DQN
    # dqn_agent = load_dqn_agent(
    #     model_folder="imitation_checkpoints",
    #     model_filename="dqn_imitation_tactical_model_final.pth",
    #     player=PLAYER_TWO,
    #     device=device,
    # )
    # ui.run_human_vs_agent(agent=dqn_agent, human_player=PLAYER_ONE)

    # 9. Human vs Imitation Tactical Depth5 DQN
    # dqn_agent = load_dqn_agent(
    #     model_folder="imitation_checkpoints",
    #     model_filename="dqn_imitation_tactical_depth5_model_final.pth",
    #     player=PLAYER_TWO,
    #     device=device,
    # )
    # ui.run_human_vs_agent(agent=dqn_agent, human_player=PLAYER_ONE)

    # 10. Human vs Imitation Tactical Depth6 DQN
    # dqn_agent = load_dqn_agent(
    #    model_folder="imitation_checkpoints",
    #    model_filename="dqn_imitation_tactical_depth6_model_final.pth",
    #    player=PLAYER_TWO,
    #    device=device,
    #)
    #tactical_agent = TacticalWrapperAgent(dqn_agent)
    #ui.run_human_vs_agent(agent=tactical_agent, human_player=PLAYER_ONE)

    # 11. Human vs Hybrid Agent
    hybrid_agent = load_hybrid_agent(
        model_folder="imitation_checkpoints",
        model_filename="dqn_imitation_tactical_depth6_model_final.pth",
        player=PLAYER_TWO,
        device=device,
        depth=4,
    )
    ui.run_human_vs_agent(agent=hybrid_agent, human_player=PLAYER_ONE)


if __name__ == "__main__":
    main()
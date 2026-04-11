import json
import os

import matplotlib.pyplot as plt


def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_json(relative_path):
    full_path = os.path.join(get_project_root(), relative_path)
    with open(full_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    matchup_names = [
        "Baseline DQN\nvs Random",
        "Double DQN\nvs Random",
        "Imitation Mixed\nvs Random",
        "Imitation Tactical D6\nvs Random",
        "Hybrid\nvs Random",
        "Hybrid\nvs Minimax D4",
        "Hybrid\nvs Minimax D6",
    ]

    result_paths = [
        "results/baseline_eval/dqn_vs_random.json",
        "results/baseline_eval/double_dqn_vs_random.json",
        "results/imitation_mixed_eval/dqn_vs_random.json",
        "results/imitation_tactical_depth6_eval/dqn_vs_random.json",
        "results/hybrid_eval_depth4/dqn_vs_random.json",
        "results/hybrid_eval_depth4/dqn_vs_minimax.json",
        "results/hybrid_eval_depth6/dqn_vs_minimax.json",
    ]

    first_player_rates = []
    second_player_rates = []
    draw_rates = []

    for path in result_paths:
        data = load_json(path)
        first_player_rates.append(data["first_player_win_rate"])
        second_player_rates.append(data["second_player_win_rate"])
        draw_rates.append(data["draw_rate"])

    x = list(range(len(matchup_names)))

    plt.figure(figsize=(12, 6))
    plt.plot(x, first_player_rates, marker="o", linewidth=2, label="First player win rate")
    plt.plot(x, second_player_rates, marker="o", linewidth=2, label="Second player win rate")
    plt.plot(x, draw_rates, marker="o", linewidth=2, label="Draw rate")

    plt.xticks(x, matchup_names, rotation=20, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("Rate")
    plt.xlabel("Matchup")
    plt.title("First-Player Advantage Across Key Matchups")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(
        get_project_root(),
        "results",
        "figures",
        "first_player_advantage.png",
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    main()
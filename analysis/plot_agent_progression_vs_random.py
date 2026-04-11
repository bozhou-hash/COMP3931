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
    agent_names = [
        "Baseline DQN",
        "Double DQN",
        "Imitation Mixed",
        "Imitation Tactical",
        "Imitation Tactical D6",
        "Hybrid",
    ]

    dqn_vs_random_paths = [
        "results/baseline_eval/dqn_vs_random.json",
        "results/baseline_eval/double_dqn_vs_random.json",
        "results/imitation_mixed_eval/dqn_vs_random.json",
        "results/imitation_tactical_eval/dqn_vs_random.json",
        "results/imitation_tactical_depth6_eval/dqn_vs_random.json",
        "results/hybrid_eval_depth4/dqn_vs_random.json",
    ]

    win_rates = []
    loss_rates = []
    draw_rates = []

    for path in dqn_vs_random_paths:
        data = load_json(path)
        win_rates.append(data["agent_a_win_rate"])
        loss_rates.append(data["agent_b_win_rate"])
        draw_rates.append(data["draw_rate"])

    x = list(range(len(agent_names)))

    plt.figure(figsize=(10, 6))
    plt.plot(x, win_rates, marker="o", linewidth=2, label="Agent win rate")
    plt.plot(x, loss_rates, marker="o", linewidth=2, label="Random win rate")
    plt.plot(x, draw_rates, marker="o", linewidth=2, label="Draw rate")

    plt.xticks(x, agent_names, rotation=20, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("Rate")
    plt.xlabel("Agent")
    plt.title("Agent Progression vs Random")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(
        get_project_root(),
        "results",
        "figures",
        "agent_progression_vs_random.png",
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    main()
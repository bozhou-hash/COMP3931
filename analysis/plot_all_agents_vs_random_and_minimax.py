import os
import json
import numpy as np
import matplotlib.pyplot as plt


def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_win_rate(results_folder, filename):
    path = os.path.join(get_project_root(), "results", results_folder, filename)

    if not os.path.exists(path):
        return None

    data = load_json(path)
    return data["agent_a_win_rate"] * 100.0


def main():
    agents = [
        ("DQN", "baseline_eval"),
        ("Double DQN", "baseline_eval"),
        ("Imitation", "imitation_eval"),
        ("Imitation Mixed", "imitation_mixed_eval"),
        ("Imitation Tactical", "imitation_tactical_eval"),
        ("Hybrid", "hybrid_eval_depth4"),
    ]

    random_rates = []
    minimax_rates = []
    labels = []

    for agent_name, folder in agents:

        # Double DQN uses different filenames
        if agent_name == "Double DQN":
            random_file = "double_dqn_vs_random.json"
            minimax_file = "double_dqn_vs_minimax.json"
        else:
            random_file = "dqn_vs_random.json"
            minimax_file = "dqn_vs_minimax.json"

        random_rate = extract_win_rate(folder, random_file)
        minimax_rate = extract_win_rate(folder, minimax_file)

        if random_rate is not None and minimax_rate is not None:
            labels.append(agent_name)
            random_rates.append(random_rate)
            minimax_rates.append(minimax_rate)

    x = np.arange(len(labels))
    width = 0.38

    plt.figure(figsize=(14, 7))

    bars1 = plt.bar(x - width / 2, random_rates, width, label="vs Random")
    bars2 = plt.bar(x + width / 2, minimax_rates, width, label="vs Minimax (Depth 4)")

    plt.title("Win Rate of All Agents vs Random and Minimax")
    plt.ylabel("Win Rate (%)")
    plt.xlabel("Agent")
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylim(0, 105)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    # value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height + 1,
                f"{height:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()

    save_path = os.path.join(
        get_project_root(),
        "results",
        "figures",
        "all_agents_vs_random_and_minimax_bar_chart.png"
    )

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved chart to: {save_path}")


if __name__ == "__main__":
    main()
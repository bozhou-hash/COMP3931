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
    models = [
        "Baseline DQN",
        "Double DQN",
        "Imitation Mixed",
        "Tactical Depth6",
        "Hybrid",
    ]

    colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
    ]

    random_files = [
        "results/baseline_eval/dqn_vs_random.json",
        "results/baseline_eval/double_dqn_vs_random.json",
        "results/imitation_mixed_eval/dqn_vs_random.json",
        "results/imitation_tactical_depth6_eval/dqn_vs_random.json",
        "results/hybrid_eval_depth6/dqn_vs_random.json",
    ]

    minimax_files = [
        "results/baseline_eval/dqn_vs_minimax.json",
        "results/baseline_eval/double_dqn_vs_minimax.json",
        "results/imitation_mixed_eval/dqn_vs_minimax.json",
        "results/imitation_tactical_depth6_eval/dqn_vs_minimax.json",
        "results/hybrid_eval_depth6/dqn_vs_minimax.json",
    ]

    jitters = [
        (-0.003, 0.000),
        (-0.001, 0.000),
        (0.001, 0.000),
        (0.003, 0.000),
        (0.000, 0.000),
    ]

    plt.figure(figsize=(7, 6))

    for model, color, r_file, m_file, (jx, jy) in zip(
        models, colors, random_files, minimax_files, jitters
    ):
        r_data = load_json(r_file)
        m_data = load_json(m_file)

        random_win = r_data["agent_a_win_rate"] + jx
        minimax_draw = m_data["draw_rate"] + jy

        plt.scatter(
            random_win,
            minimax_draw,
            s=140,
            color=color,
            edgecolors="black",
            linewidths=0.8,
            label=model,
        )

    plt.xlabel("Win rate vs Random")
    plt.ylabel("Draw rate vs Minimax")
    plt.title("Random Strength vs Minimax Draw Rate")

    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output = os.path.join(
        get_project_root(),
        "results",
        "figures",
        "random_vs_minimax_draw_scatter.png",
    )

    os.makedirs(os.path.dirname(output), exist_ok=True)
    plt.savefig(output, dpi=300)
    plt.close()

    print("Saved:", output)


if __name__ == "__main__":
    main()
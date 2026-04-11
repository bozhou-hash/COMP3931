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
    labels = [
        "vs Random",
        "vs Double DQN",
        "vs Minimax D3",
        "vs Minimax D4",
        "vs Minimax D5",
        "vs Minimax D6",
    ]

    files = [
        "results/hybrid_eval_depth4/dqn_vs_random.json",
        "results/hybrid_eval_depth4/double_dqn_vs_dqn.json",
        "results/hybrid_eval_depth3/dqn_vs_minimax.json",
        "results/hybrid_eval_depth4/dqn_vs_minimax.json",
        "results/hybrid_eval_depth5/dqn_vs_minimax.json",
        "results/hybrid_eval_depth6/dqn_vs_minimax.json",
    ]

    win_rates = []
    draw_rates = []
    loss_rates = []

    for f in files:
        data = load_json(f)

        win = data["agent_a_win_rate"]
        draw = data["draw_rate"]
        loss = data["agent_b_win_rate"]

        win_rates.append(win)
        draw_rates.append(draw)
        loss_rates.append(loss)

    x = range(len(labels))

    plt.figure(figsize=(10, 5))

    plt.plot(x, win_rates, marker="o", linewidth=2, label="Hybrid win rate")
    plt.plot(x, draw_rates, marker="o", linewidth=2, label="Draw rate")
    plt.plot(x, loss_rates, marker="o", linewidth=2, label="Loss rate")

    plt.xticks(x, labels)
    plt.ylim(0, 1.05)
    plt.ylabel("Rate")
    plt.title("Hybrid Agent Performance Across Opponents")
    plt.legend()
    plt.grid(True, alpha=0.3)

    output = os.path.join(
        get_project_root(),
        "results",
        "figures",
        "hybrid_benchmark_summary.png",
    )

    os.makedirs(os.path.dirname(output), exist_ok=True)
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved:", output)


if __name__ == "__main__":
    main()
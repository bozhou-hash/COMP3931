import json
import os
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


def load_results(folder):
    path = os.path.join(RESULTS_DIR, folder, "dqn_vs_minimax.json")
    with open(path, "r") as f:
        return json.load(f)


def main():
    depths = [3, 4, 5, 6]

    hybrid_win_rates = []
    minimax_win_rates = []
    draw_rates = []

    for d in depths:
        folder = f"hybrid_eval_depth{d}"
        results = load_results(folder)

        hybrid_win_rates.append(results["agent_a_win_rate"])
        minimax_win_rates.append(results["agent_b_win_rate"])
        draw_rates.append(results["draw_rate"])

    plt.figure(figsize=(8, 5))

    plt.plot(depths, hybrid_win_rates, marker="o", label="Hybrid win rate")
    plt.plot(depths, minimax_win_rates, marker="o", label="Minimax win rate")
    plt.plot(depths, draw_rates, marker="o", label="Draw rate")

    plt.xlabel("Minimax Depth")
    plt.ylabel("Rate")
    plt.title("Hybrid Agent vs Minimax at Different Depths")

    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()

    output_path = os.path.join(RESULTS_DIR, "hybrid_depth_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")

    print(f"Saved figure to: {output_path}")

    plt.show()


if __name__ == "__main__":
    main()
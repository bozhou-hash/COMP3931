import json
import os

import matplotlib.pyplot as plt
import numpy as np


def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_results_base_dir():
    return os.path.join(get_project_root(), "results")


def get_figures_dir():
    figures_dir = os.path.join(get_results_base_dir(), "figures", "experiment_comparison")
    os.makedirs(figures_dir, exist_ok=True)
    return figures_dir


def load_json(results_subdir, filename):
    file_path = os.path.join(get_results_base_dir(), results_subdir, filename)
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_experiment_results():
    experiments = {
        "Baseline": "baseline_eval",
        "Random Only": "curriculum_random_eval",
        "Random + Self-Play": "curriculum_selfplay_eval",
        "Full Curriculum": "curriculum_full_eval",
    }

    results = {}
    for experiment_name, subdir in experiments.items():
        results[experiment_name] = {
            "dqn_vs_random": load_json(subdir, "dqn_vs_random.json"),
            "dqn_vs_minimax": load_json(subdir, "dqn_vs_minimax.json"),
            "double_dqn_vs_dqn": load_json(subdir, "double_dqn_vs_dqn.json"),
        }

    return results


def plot_dqn_vs_random(results):
    experiments = list(results.keys())

    dqn_win_rates = [results[e]["dqn_vs_random"]["agent_a_win_rate"] * 100 for e in experiments]
    random_win_rates = [results[e]["dqn_vs_random"]["agent_b_win_rate"] * 100 for e in experiments]
    draw_rates = [results[e]["dqn_vs_random"]["draw_rate"] * 100 for e in experiments]

    x = np.arange(len(experiments))
    width = 0.25

    plt.figure(figsize=(12, 6))
    plt.bar(x - width, dqn_win_rates, width, label="DQN Win Rate")
    plt.bar(x, random_win_rates, width, label="Random Win Rate")
    plt.bar(x + width, draw_rates, width, label="Draw Rate")

    plt.xticks(x, experiments, rotation=20, ha="right")
    plt.ylabel("Percentage")
    plt.title("DQN vs Random Across Training Setups")
    plt.ylim(0, 100)
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(get_figures_dir(), "dqn_vs_random_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


def plot_dqn_vs_minimax(results):
    experiments = list(results.keys())

    dqn_win_rates = [results[e]["dqn_vs_minimax"]["agent_a_win_rate"] * 100 for e in experiments]
    minimax_win_rates = [results[e]["dqn_vs_minimax"]["agent_b_win_rate"] * 100 for e in experiments]
    draw_rates = [results[e]["dqn_vs_minimax"]["draw_rate"] * 100 for e in experiments]

    x = np.arange(len(experiments))
    width = 0.25

    plt.figure(figsize=(12, 6))
    plt.bar(x - width, dqn_win_rates, width, label="DQN Win Rate")
    plt.bar(x, minimax_win_rates, width, label="Minimax Win Rate")
    plt.bar(x + width, draw_rates, width, label="Draw Rate")

    plt.xticks(x, experiments, rotation=20, ha="right")
    plt.ylabel("Percentage")
    plt.title("DQN vs Minimax Across Training Setups")
    plt.ylim(0, 100)
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(get_figures_dir(), "dqn_vs_minimax_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


def plot_dqn_vs_double_dqn(results):
    experiments = list(results.keys())

    double_dqn_win_rates = [results[e]["double_dqn_vs_dqn"]["agent_a_win_rate"] * 100 for e in experiments]
    dqn_win_rates = [results[e]["double_dqn_vs_dqn"]["agent_b_win_rate"] * 100 for e in experiments]
    draw_rates = [results[e]["double_dqn_vs_dqn"]["draw_rate"] * 100 for e in experiments]

    x = np.arange(len(experiments))
    width = 0.25

    plt.figure(figsize=(12, 6))
    plt.bar(x - width, double_dqn_win_rates, width, label="Double DQN Win Rate")
    plt.bar(x, dqn_win_rates, width, label="DQN Win Rate")
    plt.bar(x + width, draw_rates, width, label="Draw Rate")

    plt.xticks(x, experiments, rotation=20, ha="right")
    plt.ylabel("Percentage")
    plt.title("Double DQN vs DQN Across Training Setups")
    plt.ylim(0, 100)
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(get_figures_dir(), "double_dqn_vs_dqn_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


def plot_turn_order_for_dqn_vs_random(results):
    experiments = list(results.keys())

    first_player_win_rates = [results[e]["dqn_vs_random"]["first_player_win_rate"] * 100 for e in experiments]
    second_player_win_rates = [results[e]["dqn_vs_random"]["second_player_win_rate"] * 100 for e in experiments]
    draw_rates = [results[e]["dqn_vs_random"]["draw_rate"] * 100 for e in experiments]

    x = np.arange(len(experiments))
    width = 0.25

    plt.figure(figsize=(12, 6))
    plt.bar(x - width, first_player_win_rates, width, label="First Player Win Rate")
    plt.bar(x, second_player_win_rates, width, label="Second Player Win Rate")
    plt.bar(x + width, draw_rates, width, label="Draw Rate")

    plt.xticks(x, experiments, rotation=20, ha="right")
    plt.ylabel("Percentage")
    plt.title("Turn Order in DQN vs Random Across Training Setups")
    plt.ylim(0, 100)
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(get_figures_dir(), "dqn_vs_random_turn_order_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


def print_summary(results):
    print("\n=== EXPERIMENT SUMMARY ===")
    for experiment_name, experiment_results in results.items():
        dqn_vs_random = experiment_results["dqn_vs_random"]
        dqn_vs_minimax = experiment_results["dqn_vs_minimax"]
        ddqn_vs_dqn = experiment_results["double_dqn_vs_dqn"]

        print(f"\n{experiment_name}")
        print(
            f"  DQN vs Random:   "
            f"DQN {dqn_vs_random['agent_a_win_rate']:.2%}, "
            f"Random {dqn_vs_random['agent_b_win_rate']:.2%}, "
            f"Draw {dqn_vs_random['draw_rate']:.2%}"
        )
        print(
            f"  DQN vs Minimax:  "
            f"DQN {dqn_vs_minimax['agent_a_win_rate']:.2%}, "
            f"Minimax {dqn_vs_minimax['agent_b_win_rate']:.2%}, "
            f"Draw {dqn_vs_minimax['draw_rate']:.2%}"
        )
        print(
            f"  Double DQN vs DQN: "
            f"Double DQN {ddqn_vs_dqn['agent_a_win_rate']:.2%}, "
            f"DQN {ddqn_vs_dqn['agent_b_win_rate']:.2%}, "
            f"Draw {ddqn_vs_dqn['draw_rate']:.2%}"
        )


def main():
    results = load_experiment_results()
    print_summary(results)

    plot_dqn_vs_random(results)
    plot_dqn_vs_minimax(results)
    plot_dqn_vs_double_dqn(results)
    plot_turn_order_for_dqn_vs_random(results)

    print("\nAll experiment comparison figures generated successfully.")


if __name__ == "__main__":
    main()
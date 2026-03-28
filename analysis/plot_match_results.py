import json
import os

import matplotlib.pyplot as plt
import numpy as np


def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_results_dir():
    return os.path.join(get_project_root(), "results")


def get_figures_dir():
    figures_dir = os.path.join(get_results_dir(), "figures")
    os.makedirs(figures_dir, exist_ok=True)
    return figures_dir


def load_json(filename):
    file_path = os.path.join(get_results_dir(), filename)
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_all_results():
    return {
        "DQN vs Random": load_json("dqn_vs_random.json"),
        "Double DQN vs Random": load_json("double_dqn_vs_random.json"),
        "DQN vs Minimax": load_json("dqn_vs_minimax.json"),
        "Double DQN vs Minimax": load_json("double_dqn_vs_minimax.json"),
        "Double DQN vs DQN": load_json("double_dqn_vs_dqn.json"),
    }


def plot_overall_results(results):
    matchups = list(results.keys())

    agent_a_win_rates = [results[m]["agent_a_win_rate"] * 100 for m in matchups]
    agent_b_win_rates = [results[m]["agent_b_win_rate"] * 100 for m in matchups]
    draw_rates = [results[m]["draw_rate"] * 100 for m in matchups]

    x = np.arange(len(matchups))
    width = 0.25

    plt.figure(figsize=(12, 6))
    plt.bar(x - width, agent_a_win_rates, width, label="Agent A Win Rate")
    plt.bar(x, agent_b_win_rates, width, label="Agent B Win Rate")
    plt.bar(x + width, draw_rates, width, label="Draw Rate")

    plt.xticks(x, matchups, rotation=20, ha="right")
    plt.ylabel("Percentage")
    plt.title("Overall Match Results")
    plt.ylim(0, 100)
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(get_figures_dir(), "overall_match_results.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


def plot_turn_order_results(results):
    matchups = list(results.keys())

    first_player_win_rates = [results[m]["first_player_win_rate"] * 100 for m in matchups]
    second_player_win_rates = [results[m]["second_player_win_rate"] * 100 for m in matchups]
    draw_rates = [results[m]["draw_rate"] * 100 for m in matchups]

    x = np.arange(len(matchups))
    width = 0.25

    plt.figure(figsize=(12, 6))
    plt.bar(x - width, first_player_win_rates, width, label="First Player Win Rate")
    plt.bar(x, second_player_win_rates, width, label="Second Player Win Rate")
    plt.bar(x + width, draw_rates, width, label="Draw Rate")

    plt.xticks(x, matchups, rotation=20, ha="right")
    plt.ylabel("Percentage")
    plt.title("Turn Order Performance")
    plt.ylim(0, 100)
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(get_figures_dir(), "turn_order_results.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


def plot_agent_side_wins(results):
    matchups = list(results.keys())

    agent_a_first = [results[m]["agent_a_as_first_wins"] for m in matchups]
    agent_a_second = [results[m]["agent_a_as_second_wins"] for m in matchups]
    agent_b_first = [results[m]["agent_b_as_first_wins"] for m in matchups]
    agent_b_second = [results[m]["agent_b_as_second_wins"] for m in matchups]

    x = np.arange(len(matchups))
    width = 0.2

    plt.figure(figsize=(13, 6))
    plt.bar(x - 1.5 * width, agent_a_first, width, label="Agent A Wins as First")
    plt.bar(x - 0.5 * width, agent_a_second, width, label="Agent A Wins as Second")
    plt.bar(x + 0.5 * width, agent_b_first, width, label="Agent B Wins as First")
    plt.bar(x + 1.5 * width, agent_b_second, width, label="Agent B Wins as Second")

    plt.xticks(x, matchups, rotation=20, ha="right")
    plt.ylabel("Number of Wins")
    plt.title("Wins by Side and Agent")
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(get_figures_dir(), "agent_side_wins.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


def print_summary(results):
    print("\n=== SUMMARY ===")
    for matchup, data in results.items():
        print(f"\n{matchup}")
        print(f"  Agent A win rate: {data['agent_a_win_rate']:.2%}")
        print(f"  Agent B win rate: {data['agent_b_win_rate']:.2%}")
        print(f"  Draw rate:        {data['draw_rate']:.2%}")
        print(f"  First player win rate:  {data['first_player_win_rate']:.2%}")
        print(f"  Second player win rate: {data['second_player_win_rate']:.2%}")


def main():
    results = load_all_results()
    print_summary(results)
    plot_overall_results(results)
    plot_turn_order_results(results)
    plot_agent_side_wins(results)
    print("\nAll figures generated successfully.")


if __name__ == "__main__":
    main()
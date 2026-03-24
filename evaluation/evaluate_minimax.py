from config import PLAYER_ONE, PLAYER_TWO
from env.connect4_env import Connect4Env
from agents.random_agent import RandomAgent
from agents.minimax_agent import MinimaxAgent


def play_game(env, agent_one, agent_two):
    env.reset()

    while not env.game_over:
        current_player = env.get_current_player()

        if current_player == agent_one.player:
            action = agent_one.select_action(env)
        else:
            action = agent_two.select_action(env)

        if action is None:
            break

        env.drop_piece(action)

    return env.get_result()


def evaluate_minimax(num_games=100, depth=4):
    env = Connect4Env()

    results = {
        "num_games": num_games,
        "depth": depth,
        "minimax_wins": 0,
        "random_wins": 0,
        "draws": 0,
        "minimax_as_player_one_wins": 0,
        "minimax_as_player_two_wins": 0,
    }

    for game_index in range(num_games):
        if game_index % 2 == 0:
            minimax_agent = MinimaxAgent(player=PLAYER_ONE, depth=depth)
            random_agent = RandomAgent(player=PLAYER_TWO)
        else:
            random_agent = RandomAgent(player=PLAYER_ONE)
            minimax_agent = MinimaxAgent(player=PLAYER_TWO, depth=depth)

        result = play_game(env, minimax_agent, random_agent)

        if result == minimax_agent.player:
            results["minimax_wins"] += 1
            if minimax_agent.player == PLAYER_ONE:
                results["minimax_as_player_one_wins"] += 1
            else:
                results["minimax_as_player_two_wins"] += 1
        elif result == random_agent.player:
            results["random_wins"] += 1
        else:
            results["draws"] += 1

    results["minimax_win_rate"] = results["minimax_wins"] / num_games
    results["random_win_rate"] = results["random_wins"] / num_games
    results["draw_rate"] = results["draws"] / num_games

    return results


def print_evaluation_results(results):
    print("\n=== Minimax Evaluation Results ===")
    print(f"Games played: {results['num_games']}")
    print(f"Minimax depth: {results['depth']}")
    print(f"Minimax wins: {results['minimax_wins']}")
    print(f"Random wins: {results['random_wins']}")
    print(f"Draws: {results['draws']}")
    print(f"Minimax wins as PLAYER_ONE: {results['minimax_as_player_one_wins']}")
    print(f"Minimax wins as PLAYER_TWO: {results['minimax_as_player_two_wins']}")
    print(f"Minimax win rate: {results['minimax_win_rate']:.2%}")
    print(f"Random win rate: {results['random_win_rate']:.2%}")
    print(f"Draw rate: {results['draw_rate']:.2%}")


def main():
    results = evaluate_minimax(num_games=100, depth=4)
    print_evaluation_results(results)


if __name__ == "__main__":
    main()
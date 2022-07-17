from datetime import datetime

import leg_engine
import chess.engine
import chess.pgn


def self_play():
    # opening_book_file_path=None,
    leg = leg_engine.LegEngine(use_endgame_tables=True,
                               max_simulations=70000, use_random_opening=True, simulation_miss_max=20)

    event_timestamp = datetime.now()
    white_wins = 0.0
    black_wins = 0.0
    for match_round in range(1, 3001):
        game = chess.pgn.Game()
        node = game
        current_board = chess.Board()
        print(f"\nRound: {match_round} Ply: {current_board.ply()} wins: {white_wins:g}-{black_wins:g}\n{current_board}")
        while not current_board.is_game_over():
            next_move = leg.find_best_move(current_board.fen(), 1, 2)  # 6, 60)  # 2, 0.5) #  1, 0.5)
            current_board.push(next_move)
            node = node.add_variation(next_move)
            print(
                f"\nRound: {match_round} Ply: {current_board.ply()} wins: {white_wins:g}-{black_wins:g}\n{current_board}")
        o = current_board.outcome()
        if o.winner is None:
            white_wins += 0.5
            black_wins += 0.5
        elif o.winner:
            white_wins += 1.0
        else:
            black_wins += 1.0
        game.headers["Event"] = "Fast Self Training"
        game.headers["Site"] = "Local"
        game.headers["Date"] = "2022.05.22"
        game.headers["Result"] = current_board.result()
        game.headers["White"] = "LEG Engine"
        game.headers["Black"] = "LEG Engine"
        game.headers["Round"] = str(match_round)
        print(f'\n{current_board}')
        print("{} by {}".format(o.result(), o.termination.name))
        print(game)
        date_time = event_timestamp.strftime("%Y_%m_%d_%H_%M_%S")
        print(game, file=open(
            f"training_files\\poor_generated_training_files\\leg_engine_{date_time}.pgn",
            "a"), end="\n\n")
        print(f"\nWins: {white_wins:g}-{black_wins:g}")


if __name__ == '__main__':
    self_play()

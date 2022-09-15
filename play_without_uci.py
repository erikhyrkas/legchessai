from datetime import datetime

import leg_engine
import chess.engine
import chess.pgn


def play_stockfish():
    # opening_book_file_path=None,
    leg = leg_engine.LegEngine(max_simulations=70000, use_random_opening=False,
                               simulation_miss_max=20, use_endgame_tables=False)

    event_timestamp = datetime.now()
    white_wins = 0.0
    black_wins = 0.0
    white_accuracy_correct = 0
    white_accuracy_total = 0
    with chess.engine.SimpleEngine.popen_uci(r"stockfish\\stockfish_15_x64_avx2.exe") as stockfish_analyzer:
        # with chess.engine.SimpleEngine.popen_uci(r"C:\\Program Files (x86)\\ShredderChess\\Deep Shredder 13\\EngineDeepShredder13UCIx64.exe") as opponent_engine:
        with chess.engine.SimpleEngine.popen_uci(r"stockfish\\stockfish_15_x64_avx2.exe") as opponent_engine:
            opponent_engine.configure({"UCI_LimitStrength": True})
            opponent_engine.configure({"UCI_Elo": 1550})
            # with 20 million positions trained
            # depth  1, 1550 wins 54% of the time over 100 games
            # depth  2, 1550 wins 55.5% of the time over 100 games
            # depth  3, 1550 wins 52% of the time over 100 games
            # depth  4, 1550 wins 55% of the time over 100 games
            # depth 14, 1550 wins 44% of the time over 25 games
            for match_round in range(1, 101):
                game = chess.pgn.Game()
                node = game
                # board = game.board()
                current_board = chess.Board()
                print(f"\n{current_board}")
                while not current_board.is_game_over():
                    next_move = None
                    print(f"Round: {match_round} Ply: {current_board.ply()} wins: {white_wins:g}-{black_wins:g}")
                    if current_board.turn == chess.WHITE:
                        # print("LEG's turn starts. Thinking...")
                        next_move = leg.find_best_move(current_board.fen(), 2, 300)  # 6, 60)  # 2, 0.5) #  1, 0.5)
                        print(f"Leg's move: {next_move}")
                        # real_best_move = stockfish_analyzer.play(current_board, chess.engine.Limit(time=0.1)).move
                        # print(f"Stockfish would have done: {real_best_move}")
                        # white_accuracy_total += 1
                        # if next_move == real_best_move:
                        #     white_accuracy_correct += 1
                        # white_accuracy = (white_accuracy_correct*100)/white_accuracy_total
                        # print(f"White Accuracy: {white_accuracy:.1f}%")
                        # sf_score = stockfish_analyzer.analyse(current_board, chess.engine.Limit(time=0.1))['score']
                        # print(f"Analysis pre-move: {sf_score}")
                    else:
                        # print("Stockfish's turn starts:")
                        next_move = opponent_engine.play(current_board, chess.engine.Limit(time=0.1)).move
                        print(f"Stockfish's move: {next_move}")
                    current_board.push(next_move)
                    node = node.add_variation(next_move)
                    print(f"{current_board}")
                    # print(f"{current_board.fen()}")
                    # sf_score = stockfish_analyzer.analyse(current_board, chess.engine.Limit(time=0.1))['score']
                    # print(f"Analysis post-move: {sf_score}")
                    # leval = leg_engine.eval_for_current_player(current_board)
                    # if current_board.turn == chess.BLACK:
                    #     print(f"Leg eval for Black: {leval:.4f}")
                    # else:
                    #     print(f"Leg eval for White: {leval:.4f}")
                    # print()
                    # input('ENTER for next move> ')

                o = current_board.outcome()
                if o.winner is None:
                    white_wins += 0.5
                    black_wins += 0.5
                elif o.winner:
                    white_wins += 1.0
                else:
                    black_wins += 1.0
                game.headers["Event"] = "Exhibition"
                game.headers["Site"] = "Local"
                game.headers["Date"] = "2022.05.22"
                game.headers["Result"] = current_board.result()
                game.headers["White"] = "LEG Engine"
                game.headers["Black"] = "Stockfish 15"
                game.headers["Round"] = str(match_round)
                print(f'\n{current_board}')
                print("{} by {}".format(o.result(), o.termination.name))
                print(game)
                date_time = event_timestamp.strftime("%Y_%m_%d_%H_%M_%S")
                print(game, file=open(
                    f"training_files\\poor_generated_training_files\\leg_engine_{date_time}.pgn",
                    "a"), end="\n\n")
                print(f"Round: {match_round} Ply: {current_board.ply()} wins: {white_wins:g}-{black_wins:g}")


if __name__ == '__main__':
    play_stockfish()

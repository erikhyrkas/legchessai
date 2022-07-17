from datetime import datetime

import chess
import chess.engine
import chess.pgn


def play(event_timestamp, white_engine, black_engine, game_round, white_label, black_label,
         timelimit_white, timelimit_black, folder):
    game = chess.pgn.Game()
    node = game
    board = game.board()

    print(f'\n{board}')
    while not board.is_game_over():
        if board.turn == chess.WHITE:
            result = white_engine.play(board, chess.engine.Limit(time=timelimit_white))
        else:
            result = black_engine.play(board, chess.engine.Limit(time=timelimit_black))
        move = result.move
        board.push(move)
        node = node.add_variation(move)
        # print(f'\n{move}\n{board}')
        print(f'\n{board}')

    o = board.outcome()
    game.headers["Event"] = "Training"
    game.headers["Site"] = "Local"
    game.headers["Date"] = "2022.05.22"
    game.headers["Result"] = board.result()
    game.headers["White"] = white_label
    game.headers["Black"] = black_label
    game.headers["Round"] = str(game_round)

    print(f'\n{board}')
    print("{} by {}".format(o.result(), o.termination.name))
    print(game)
    date_time = event_timestamp.strftime("%Y_%m_%d_%H_%M_%S")
    outcome = board.outcome()
    if outcome.winner is None:
        outcome_string = 'draw'
    elif outcome.winner:
        outcome_string = 'white'
    else:
        outcome_string = 'black'

    print(game, file=open(
        f"{folder}\\{white_label}_vs_{black_label}_{outcome_string}_{date_time}.pgn",
        "a"), end="\n\n")


def generate_pgns_from_engines():
    # engine_white = chess.engine.SimpleEngine.popen_uci(
    #    r"C:\\Program Files (x86)\\ShredderChess\\Deep Shredder 13\\EngineDeepShredder13UCIx64.exe")
    # engine_black = chess.engine.SimpleEngine.popen_uci(
    #    r"C:\\Program Files (x86)\\ShredderChess\\Deep Shredder 13\\EngineDeepShredder13UCIx64.exe")
    engine_white = chess.engine.SimpleEngine.popen_uci(r"stockfish\\stockfish_15_x64_avx2.exe")
    engine_black = chess.engine.SimpleEngine.popen_uci(r"stockfish\\stockfish_15_x64_avx2.exe")
    timestamp = datetime.now()
    # for weak_opponent_strength in range(0, 21, 5):
    #     for game_number in range(1, 15):
    #         play(timestamp, engine_white, engine_black, game_number, 20, weak_opponent_strength,
    #              0.1, "training_files\\generated_training_files")
    #         play(timestamp, engine_white, engine_black, game_number, weak_opponent_strength, 20,
    #              0.1, "training_files\\generated_training_files")
    for weak_opponent_strength in range(0, 26):
        for game_number in range(0, 301):
            if weak_opponent_strength >= 0:
                engine_white.configure({"Skill Level": weak_opponent_strength})  # 0 - 20
                engine_black.configure({"Skill Level": weak_opponent_strength})  # 0 - 20
                # setoption name Skill Level value 0
                # setoption name Skill Level Maximum Error value 900
                # setoption name Skill Level Probability value 10
                if weak_opponent_strength < 5:
                    engine_white.configure({"Skill Level Maximum Error": 900})
                    engine_black.configure({"Skill Level Maximum Error": 900})
                    engine_white.configure({"Skill Level Probability": 10 + (weak_opponent_strength * 10)})
                    engine_black.configure({"Skill Level Probability": 10 + (weak_opponent_strength * 10)})

            play(timestamp, engine_white, engine_black, game_number, str(weak_opponent_strength),
                 str(weak_opponent_strength),
                 0.1, "training_files\\poor_generated_training_files")
            play(timestamp, engine_white, engine_black, game_number, str(weak_opponent_strength),
                 str(weak_opponent_strength),
                 0.1, "training_files\\poor_generated_training_files")
    engine_white.quit()
    engine_black.quit()


if __name__ == '__main__':
    generate_pgns_from_engines()

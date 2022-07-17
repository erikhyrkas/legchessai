import os
from datetime import datetime
from os import listdir
from os.path import isfile, join
import numpy
import chess.pgn

from stockfish import Stockfish


def create_scored_fens(show_progress=False, depth=8, threads=4):
    stockfish = Stockfish(path="stockfish\\stockfish_15_x64_avx2.exe")
    stockfish.set_depth(depth)
    stockfish.update_engine_parameters({"Threads": threads})

    training_files_path = 'training_files/combined_eval'
    training_files = [join(training_files_path, f) for f in listdir(training_files_path) if
                      isfile(join(training_files_path, f))]

    print("create_scored_fens: Loading training files:")
    print(training_files)
    scored_fens = []
    if show_progress:
        print("Progress: ")
    progress_counter = 0
    progress_max = len(training_files)
    part = 1
    records = 0
    event_timestamp = datetime.now()
    date_time = event_timestamp.strftime("%Y_%m_%d_%H_%M_%S")
    pgns_names_to_save = []
    if os.path.exists("training_files/scored_files.txt"):
        with open("training_files/scored_files.txt", "r") as scored_files_txt:
            scored_files = scored_files_txt.read().splitlines()
    else:
        scored_files = []

    print(scored_files)
    for pgn_file_name in training_files:
        if pgn_file_name in scored_files:
            if show_progress:
                progress_counter += 1
                print(f"{pgn_file_name} was already processed. Skipping.")
            continue
        if records > 2000:
            part, records = save_scored_fens(date_time, part, pgns_names_to_save, records, scored_fens, show_progress)
        pgn = open(pgn_file_name)
        next_game = chess.pgn.read_game(pgn)
        progress_counter += 1
        game_progress = 0
        while next_game is not None:
            next_game_board = next_game.board()
            if show_progress:
                game_progress += 1
                print(f"{progress_counter} of {progress_max} {pgn_file_name} Game# {game_progress}")
            else:
                print(f"{pgn_file_name}")

            records += 1
            scored_fens.append(build_scored_fen_result_from_board(next_game_board, show_progress, stockfish))
            for move in next_game.mainline_moves():
                next_game_board.push(move)
                records += 1
                scored_fens.append(build_scored_fen_result_from_board(next_game_board, show_progress, stockfish, move.uci()))

            next_game = chess.pgn.read_game(pgn)
            if show_progress:
                print('')
        pgns_names_to_save.append(pgn_file_name)
        scored_files.append(pgn_file_name)
    save_scored_fens(date_time, part, pgns_names_to_save, records, scored_fens, show_progress)
    print("create_scored_fens: done.")


def build_scored_fen_result_from_board(next_game_board, show_progress, stockfish, uci=''):
    stockfish.set_fen_position(next_game_board.fen())
    eval_obj = stockfish.get_evaluation()
    eval_val = eval_obj.get('value')
    top_moves = stockfish.get_top_moves()
    scored_fen_result = calculated_scored_fen(eval_obj, eval_val, next_game_board, show_progress,
                                              top_moves, uci)
    return scored_fen_result


def calculated_scored_fen(eval_obj, eval_val, next_game_board, show_progress, top_moves, uci):
    top_moves_strings = []
    for top_move in top_moves:
        top_move_centipawn = top_move.get('Centipawn') or '-'
        top_move_mate = top_move.get('Mate') or '-'
        top_moves_strings.append(
            f"{top_move.get('Move')}/{top_move_centipawn}/{top_move_mate}")
    top_moves_string = ';'.join(top_moves_strings)
    if next_game_board.turn:
        turn_player_name = 'White'
    else:
        turn_player_name = 'Black'
    if eval_obj.get('type') == 'cp':
        eval_type = 0
        mate_count = '-'
        pos_evaluation = int(eval_val)
    else:
        eval_type = 1
        mate_count = eval_val
        pos_evaluation = '-'
    scored_fen_result = (next_game_board.fen(), pos_evaluation, mate_count, top_moves_string)
    # print(f"{scored_fen_result}")
    if not show_progress:
        print(f"\n{turn_player_name} {eval_type}:{pos_evaluation} / {mate_count} [{eval_val}] {uci}")
        print(f"{scored_fen_result}")
        print(f"{next_game_board}")

    return scored_fen_result


def save_scored_fens(date_time, part, pgns_names_to_save, records, scored_fens, show_progress):
    if len(scored_fens) == 0:
        return part, records
    if show_progress:
        print(f"\nSaving {records}...")
    with open(f'training_files\\scored_fens\\combined_fens_{date_time}_{part}.txt', 'w') as result_file:
        for scored_fen in scored_fens:
            print(f"{scored_fen[0]}, {scored_fen[1]}, {scored_fen[2]}, {scored_fen[3]}", file=result_file)
    records = 0
    part += 1
    scored_fens.clear()
    if pgns_names_to_save is not None:
        with open("training_files/scored_files.txt", "a") as scored_files_txt:
            for pgn_name_to_save in pgns_names_to_save:
                scored_files_txt.write(f"{pgn_name_to_save}\n")
        pgns_names_to_save.clear()
    return part, records


if __name__ == '__main__':
    numpy.set_printoptions(threshold=numpy.inf)
    # create_evaluation_datasets(True, 14, 8)
    create_scored_fens(True, 8, 8)

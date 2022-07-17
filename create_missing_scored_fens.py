import os
import random
from datetime import datetime
from math import ceil
from os import listdir
from os.path import join, isfile

import chess
import numpy
from stockfish import Stockfish

from data_utils import encode_scored_fen, encoded_board_to_string, all_moves_from_encoded_array, scored_fen_to_board
from pgns_to_scored_fens import build_scored_fen_result_from_board, save_scored_fens
from scored_fen_to_dataset import file_to_randomized_lines


def create_missing_scored_fens(summarize_progress=False, depth=8, threads=8):
    stockfish = Stockfish(path="stockfish\\stockfish_15_x64_avx2.exe")
    stockfish.set_depth(depth)
    stockfish.update_engine_parameters({"Threads": threads})

    training_files_path = 'training_files/scored_fens'
    training_files = [join(training_files_path, f) for f in listdir(training_files_path) if
                      isfile(join(training_files_path, f))]

    print("scored_fen_to_dataset: Loading training files:")
    random.shuffle(training_files)
    print(training_files)
    fen_dedupe_set = set()
    fen_set = set()
    if summarize_progress:
        print("Progress: ")
    file_count = 0
    for txt_file_name in training_files:
        text_lines = file_to_randomized_lines(txt_file_name)
        random.shuffle(text_lines)
        file_count += 1
        for text_line in text_lines:
            clean_text_line = text_line.strip()
            if len(clean_text_line) == 0:
                continue
            fen_only = clean_text_line.strip().split(',')[0].strip()
            fen_no_turn = fen_only.rsplit(' ', 1)[0]
            r_split_fen_no_turn = fen_no_turn.rsplit(' ', 1)
            fen_no_half_turn_clock = r_split_fen_no_turn[0]
            half_turn_clock = int(r_split_fen_no_turn[1])
            # if half_turn_clock < 99:
            #     fen_no_half_turn_clock += ' 0'
            # else:
            #     fen_no_half_turn_clock += ' ' + half_turn_clock
            if half_turn_clock >= 99:
                # let's not worry about moves that avoid the half_turn_clock
                # It requires a lot more training to for the model to understand than the situation requires
                # in standard play.
                continue
            if fen_no_half_turn_clock in fen_dedupe_set:
                continue
            fen_dedupe_set.add(fen_no_half_turn_clock)
            if chess.Board(fen_only).is_game_over():
                # there are no moves to make after this, and we can detect these positions heuristically.
                continue
            fen_set.add(fen_only)
        if summarize_progress and (file_count % 10 == 0):
            print('.', end='')
    print("\nFilling in missing scored fens\n")
    event_timestamp = datetime.now()
    date_time = event_timestamp.strftime("%Y_%m_%d_%H_%M_%S")
    scored_fens = []
    part = 1
    records = 0
    total_len = len(fen_set)
    total_count = 0
    fen_count = 0
    for fen in fen_set:
        if total_count > 1000000:
            # on my machine, roughly 1 hour of run time for every 100,000
            break
        fen_count += 1
        if records > 50000:
            part, records = save_scored_fens(date_time, part, None, records, scored_fens, summarize_progress)
            print(f"Part {part}: {total_count} created. {fen_count} of {total_len} processed")
        next_game_board = chess.Board(fen)
        # if next_game_board.fullmove_number > 25:
        #     continue
        for legal_move in next_game_board.legal_moves:
            temp_board = next_game_board.copy()
            temp_board.push(legal_move)
            if temp_board.is_game_over():
                continue
            fen_only = temp_board.fen()
            fen_no_turn = fen_only.rsplit(' ', 1)[0]
            r_split_fen_no_turn = fen_no_turn.rsplit(' ', 1)
            fen_no_half_turn_clock = r_split_fen_no_turn[0]
            if fen_no_half_turn_clock in fen_dedupe_set:
                continue
            fen_dedupe_set.add(fen_no_half_turn_clock)
            scored_fens.append(build_scored_fen_result_from_board(temp_board, summarize_progress, stockfish))
            records += 1
            total_count += 1
            # print(f"Adding missing fen: {fen_only}")
            if summarize_progress and (records % 5000):
                print('.', end='')
    save_scored_fens(date_time, part, None, records, scored_fens, summarize_progress)
    print(f"Part {part}: {total_count} created. {fen_count} of {total_len} processed")


if __name__ == '__main__':
    numpy.set_printoptions(threshold=numpy.inf)
    create_missing_scored_fens(True)

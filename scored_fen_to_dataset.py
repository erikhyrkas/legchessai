import os
import random
from datetime import datetime
from math import ceil
from os import listdir
from os.path import join, isfile

import chess
import numpy

from data_utils import encode_scored_fen, encoded_board_to_string, all_moves_from_encoded_array, scored_fen_to_board

MAX_DATASET_FILE_SIZE = 1024


def calc_batch_size(x):
    if MAX_DATASET_FILE_SIZE % x != 0:
        raise Exception(f'{MAX_DATASET_FILE_SIZE} is not divisible by {x}.')
    return round(MAX_DATASET_FILE_SIZE / x)


def scored_fen_to_dataset(summarize_progress=False):
    training_files_path = 'training_files/scored_fens'
    training_files = [join(training_files_path, f) for f in listdir(training_files_path) if
                      isfile(join(training_files_path, f))]

    print("scored_fen_to_dataset: Loading training files:")
    random.shuffle(training_files)
    print(training_files)
    fen_dedupe_set = set()
    fen_skip_count = 0
    encoded_fens = []
    encoded_evals = []
    encoded_best_moves = []
    if summarize_progress:
        print("Progress: ")
    progress_counter = 0
    part = 1
    records = 0
    event_timestamp = datetime.now()
    date_time = event_timestamp.strftime("%Y_%m_%d_%H_%M_%S")
    txt_names_to_save = []
    if os.path.exists("training_files/scored_files_to_dataset.txt"):
        with open("training_files/scored_files_to_dataset.txt", "r") as recorded_files_txt:
            recorded_files = recorded_files_txt.read().splitlines()
    else:
        recorded_files = []
    print(f"Skipping: {recorded_files}")
    for txt_file_name in training_files:
        if txt_file_name in recorded_files:
            if summarize_progress:
                progress_counter += 1
            continue
        if records >= MAX_DATASET_FILE_SIZE:
            save_progress(date_time, encoded_best_moves, encoded_evals, encoded_fens, part, records, summarize_progress,
                          txt_names_to_save)
            records = 0
            part += 1
            encoded_fens.clear()
            encoded_evals.clear()
            encoded_best_moves.clear()
            txt_names_to_save.clear()

        text_lines = file_to_randomized_lines(txt_file_name)
        random.shuffle(text_lines)
        txt_names_to_save.append(txt_file_name)
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
                fen_skip_count += 1
                continue
            if fen_no_half_turn_clock in fen_dedupe_set:
                fen_skip_count += 1
                continue
            fen_dedupe_set.add(fen_no_half_turn_clock)
            if chess.Board(fen_only).is_game_over():
                # there are no moves to make after this, and we can detect these positions heuristically.
                fen_skip_count += 1
                continue
            next_encoded_fen, next_encoded_eval, next_encoded_best_moves, next_flipped = encode_scored_fen(
                clean_text_line)
            # if not next_flipped:  # f4d6 -> became a7a6, but should have been f5d3
            #     continue
            encoded_fens.append(next_encoded_fen)
            encoded_evals.append(next_encoded_eval)
            encoded_best_moves.append(next_encoded_best_moves)
            records += 1
            # print(clean_text_line)
            # print(scored_fen_to_board(clean_text_line))
            # print(f"turn: {scored_fen_to_board(clean_text_line).turn}")
            # print(encoded_board_to_string(next_encoded_fen))
            # print(next_encoded_eval)
            # print(list(all_moves_from_encoded_array(next_encoded_best_moves).islice(0, 1)))
            # input(">")
            # print(f"{len(next_encoded_fen)} {next_encoded_eval} {len(next_encoded_best_moves)}")
            # print(f"{next_encoded_fen}\n{next_encoded_eval}\n{next_encoded_best_moves}")
            if not summarize_progress:
                print(f"{clean_text_line}")
            if records >= MAX_DATASET_FILE_SIZE:
                save_progress(date_time, encoded_best_moves, encoded_evals, encoded_fens, part, records,
                              summarize_progress, txt_names_to_save, fen_skip_count)
                fen_skip_count = 0
                records = 0
                part += 1
                encoded_fens.clear()
                encoded_evals.clear()
                encoded_best_moves.clear()
                txt_names_to_save.clear()
        if summarize_progress:
            print('.', end='')
    print()
    if records >= MAX_DATASET_FILE_SIZE:
        save_progress(date_time, encoded_best_moves, encoded_evals, encoded_fens, part, records, summarize_progress,
                      txt_names_to_save, fen_skip_count)
    else:
        print(f"Not saving last dataset, since it was only {records} and not near our goal of {MAX_DATASET_FILE_SIZE}")
    print("scored_fen_to_dataset: done.")


def file_to_randomized_lines(txt_file_name):
    with open(txt_file_name) as scored_fen_txt:
        # I tried to dedupe, but it had virtually no effect. variations from eval?
        text_lines = list(set(scored_fen_txt.readlines()))
    return text_lines


def save_progress(date_time, encoded_best_moves, encoded_evals, encoded_fens, part, records, summarize_progress,
                  txt_names_to_save, fen_skip_count):
    if len(encoded_best_moves) == 0:
        return
    if not summarize_progress:
        print(f"\nSaving {records} (Skipped fens: {fen_skip_count}...")
    numpy.savez(f'dataset\\scored_fen_dataset_{date_time}_{part}.npz', numpy.array(encoded_fens),
                numpy.array(encoded_evals), numpy.array(encoded_best_moves))
    with open("training_files/scored_files_to_dataset.txt", "a") as recorded_files_txt:
        for txt_name_to_save in txt_names_to_save:
            recorded_files_txt.write(f"{txt_name_to_save}\n")


if __name__ == '__main__':
    numpy.set_printoptions(threshold=numpy.inf)
    scored_fen_to_dataset(True)

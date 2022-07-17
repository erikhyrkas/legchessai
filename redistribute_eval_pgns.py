import glob
import os
from datetime import datetime

import chess.pgn


def combine_and_redistribute(show_progress=False):
    training_files = []
    for filename in glob.iglob('training_files\\**\\*.pgn', recursive=True):
        training_files.append(filename)

    print("combine_and_redistribute: Loading training files:")
    print(training_files)

    if show_progress:
        print("Progress: ")
    progress_counter = 0
    progress_max = len(training_files)
    part = 1
    records = 0
    event_timestamp = datetime.now()
    date_time = event_timestamp.strftime("%Y_%m_%d_%H_%M_%S")
    if os.path.exists("training_files/eval_combined_recorded_files.txt"):
        with open("training_files/eval_combined_recorded_files.txt", "r") as recorded_files_txt:
            recorded_files = recorded_files_txt.read().splitlines()
    else:
        recorded_files = []

    print(recorded_files)
    for pgn_file_name in training_files:
        if 'combined_' in pgn_file_name:
            continue
        if pgn_file_name in recorded_files:
            if show_progress:
                print(f"{pgn_file_name} was already processed.Skipping.")
            continue
        with open(pgn_file_name, encoding="utf-8-sig") as pgn:
            game = chess.pgn.read_game(pgn)
            game_progress = 0
            progress_counter += 1
            while game is not None:
                records += 1
                game_progress += 1
                if show_progress:
                    print(f"{progress_counter} of {progress_max} {pgn_file_name} Game# {game_progress} Part# {part} Record# {records}")
                if records > 499:
                    records = 0
                    part += 1
                print(game, file=open(
                    f"training_files\\combined_eval\\combined_eval_{date_time}_{part}.pgn",
                    "a"), end="\n\n")
                game = chess.pgn.read_game(pgn)
        with open("training_files/eval_combined_recorded_files.txt", "a") as recorded_files_txt:
            recorded_files_txt.write(f"{pgn_file_name}\n")

    print("combine_and_redistribute: done.")


if __name__ == '__main__':
    combine_and_redistribute(True)

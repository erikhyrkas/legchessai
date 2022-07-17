from create_missing_scored_fens import create_missing_scored_fens
from generate_pgn_from_engine import generate_pgns_from_engines
from pgns_to_scored_fens import create_scored_fens
from play_self_for_training import self_play
from play_without_uci import play_stockfish
from redistribute_eval_pgns import combine_and_redistribute
from scan_dataset import scan_dateset_main
from scored_fen_to_dataset import calc_batch_size, scored_fen_to_dataset
from train_model import train
from tensorflow.keras.regularizers import l2
import os
import glob


def prep():
    # combine_and_redistribute(True)
    # create_scored_fens(True, 8, 8)
    files = glob.glob('dataset/*.npz')
    for f in files:
        os.remove(f)
    try:
        os.remove('training_files/scored_files_to_dataset.txt')
    except FileNotFoundError:
        pass
    scored_fen_to_dataset(True)


def perform_training():
    # disable use of GPU:
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    train(calc_batch_size(256), 0.98, None, 23, 2, False, l2(1e-3), True,
          "categorical_crossentropy", "categorical_crossentropy", 1.0, 1.0, 1e-5, 80, None, 256)
    # scan_dateset_main()  # needs to use most recent model


def create_more_data():
    # Very first run, if you have no data, you should start here:
    generate_pgns_from_engines()
    # Once you have data, it's good to exercise the moves you make to learn from them:
    # self_play()
    # and see how stockfish would respond to what you know:
    # play_stockfish()
    # let's make we use what we are about to build on:
    combine_and_redistribute(True)
    create_scored_fens(True, 8, 8)
    # let's flesh out what we have with even more options:
    for i in range(10):
        create_missing_scored_fens(True)


if __name__ == '__main__':
    # create_more_data()
    # prep()
    perform_training()

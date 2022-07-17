import glob
import random

import numpy

import leg_engine
from data_utils import all_moves_from_encoded_array, encoded_board_to_string, encoded_board_to_simple_eval, \
    NUMBER_OF_TOP_MOVES_TO_INCLUDE, category_to_eval, confidence_of_eval, category_to_confidence_winning, \
    trim_top_moves, TOP_MOVE_BEST_TO_WORST_SCORE_RATIO
from scored_fen_to_dataset import MAX_DATASET_FILE_SIZE
from train_model import MultiFileEvalGenerator


def scan_dateset_main(limit_files=None):
    numpy.set_printoptions(threshold=numpy.inf)
    training_files = []
    for filename in glob.iglob(f'dataset\\scored_fen_dataset_*.npz', recursive=False):
        training_files.append(filename)
        if (limit_files is not None) and (len(training_files) >= limit_files):
            break
    random.shuffle(training_files)
    training_generator = MultiFileEvalGenerator(training_files, False,
                                                MAX_DATASET_FILE_SIZE)  # 200 gives 300 records per batch
    # check_score(training_generator)
    # check_best_moves(training_generator)
    # walk_results(training_generator)
    sample_results(training_generator, 1500)


def check_best_moves(training_generator):
    full_set = set()
    for x in training_generator:
        # print(f"{len(x[1][0])}")
        # print(len(x[1][1][0]))
        # print(f"{len(x[1][1])}")
        # print(f"{x[1][1]}")
        white_moves = all_moves_from_encoded_array(x[1][1][0], True).islice(0, 5)
        black_moves = all_moves_from_encoded_array(x[1][1][0], False).islice(0, 5)
        for move in white_moves:
            full_set.add(move[0])
        for move in black_moves:
            full_set.add(move[0])
    print(full_set)


def check_score(training_generator):
    eval_count = 0
    summed_evals = 0
    min_eval = 100
    max_eval = -100
    black_non_mate_results = 0
    white_non_mate_results = 0
    for x in training_generator:
        # print(f"{len(x[1][0])}")
        # print(x[0][2])
        for eval_result in x[1][0]:
            # if eval_result > 0.5:
            #     eval_result = 1.0
            # else:
            #     eval_result = 0.0
            # eval_result = round(eval_result)
            eval_count += 1
            summed_evals += eval_result
            if (eval_result < min_eval) and (eval_result > 0.0):
                min_eval = eval_result
            if (eval_result > max_eval) and (eval_result < 1.0):
                max_eval = eval_result
            if (eval_result > 0.5) and (eval_result < 1.0):
                white_non_mate_results += 1
            if (eval_result > 0.0) and (eval_result < 0.5):
                black_non_mate_results += 1
            # print(eval_result)
    print(f"average: {summed_evals} / {eval_count} = {summed_evals / eval_count}")
    print(f"min: {min_eval} max: {max_eval}")
    print(f"black imperfect results: {black_non_mate_results} white imperfect results: {white_non_mate_results}")


def walk_results(training_generator):
    legengine = leg_engine.LegEngine(max_simulations=70000)
    total_scanned = 0
    total_good_predictions = 0
    total_best_move_ranks = 0
    total_best_moves = 0
    total_hard_predictions = 0
    total_confident_good_predictions = 0
    average_confidence_while_winning_sum = 0
    average_confidence_while_winning_count = 0
    average_confidence_while_losing_sum = 0
    average_confidence_while_losing_count = 0
    numpy.set_printoptions(threshold=numpy.inf)
    for x in training_generator:
        encoded_board = x[0][0]
        one_record = x[1]
        pos_eval = category_to_eval(one_record[0][0])
        best_moves = one_record[1][0]
        all_best_moves = list(all_moves_from_encoded_array(best_moves).islice(0, 1))[0]
        simple_pos_eval = encoded_board_to_simple_eval(encoded_board)
        expanded_dims = numpy.expand_dims(encoded_board, 0)
        leg_predicted_eval, leg_encoded_top_moves = legengine.ai_model.predict(expanded_dims, verbose=0)
        leg_all_best_moves = trim_top_moves(
            list(all_moves_from_encoded_array(leg_encoded_top_moves[0]).islice(0, NUMBER_OF_TOP_MOVES_TO_INCLUDE)))
        confidence_of_winning = category_to_confidence_winning(leg_predicted_eval[0])
        if pos_eval < 0.5:
            average_confidence_while_losing_count += 1
            average_confidence_while_losing_sum += confidence_of_winning
        else:
            average_confidence_while_winning_count += 1
            average_confidence_while_winning_sum += confidence_of_winning
        simple_leg_predicted_eval = category_to_eval(leg_predicted_eval[0])
        simple_leg_predicted_eval_confidence = confidence_of_eval(leg_predicted_eval[0])
        prediction_diff = abs(pos_eval - simple_leg_predicted_eval)
        has_best_move = False
        best_move_score = None
        best_move_rank = None
        current_rank = 0
        for next_move in leg_all_best_moves:
            current_rank += 1
            if all_best_moves[0].uci() == next_move[0].uci():
                has_best_move = True
                best_move_score = next_move[1]
                best_move_rank = current_rank
                break
        if abs(pos_eval - simple_pos_eval) > 0.02:
            total_hard_predictions += 1
        print(encoded_board_to_string(encoded_board))
        print(
            f"stockfish eval: {pos_eval:.4f} / simple piece value: {simple_pos_eval:.4f} / best moves: {all_best_moves}")
        print(
            f"LEG eval: {simple_leg_predicted_eval:.4f} {leg_predicted_eval[0]} ({prediction_diff:.4f}) / LEG best moves: {leg_all_best_moves}")
        found_val = ''
        if has_best_move:
            total_best_moves += 1
            total_best_move_ranks += best_move_rank
            found_val = f'/ Best move found: {all_best_moves[0].uci()} ({best_move_score:.4f})'
        good_val = ''
        if prediction_diff < 0.01:
            good_val = 'Good Prediction! '
            total_good_predictions += 1
            if simple_leg_predicted_eval_confidence > 0.5:
                total_confident_good_predictions += 1
        print(f"\n{good_val}{prediction_diff:.4f} (confidence: {simple_leg_predicted_eval_confidence:.4f}) {found_val}")
        total_scanned += 1
        best_move_percent = int((total_best_moves / total_scanned) * 100)
        if total_best_moves > 0:
            best_move_rank_avg = int(total_best_move_ranks / total_best_moves)
        else:
            best_move_rank_avg = 0
        prediction_percent = int((total_good_predictions / total_scanned) * 100)
        if average_confidence_while_winning_count > 0:
            winner_conf_percent = int(
                (average_confidence_while_winning_sum / average_confidence_while_winning_count) * 100)
        else:
            winner_conf_percent = 0
        if average_confidence_while_losing_count > 0:
            loser_conf_percent = int(
                (average_confidence_while_losing_sum / average_confidence_while_losing_count) * 100)
        else:
            loser_conf_percent = 0
        result = input(
            f'Hit Enter for next or q to quit (moves: {total_best_moves}/{total_scanned} {best_move_percent}% {best_move_rank_avg} avg rank'
            f' predictions: {total_good_predictions}/{total_scanned} {prediction_percent}%'
            f' wcp: {winner_conf_percent}% lcp: {loser_conf_percent}%'
            f' - difficult predictions: {total_hard_predictions} - confident good predictions: {total_confident_good_predictions})> ')
        if 'q' == result:
            break


def sample_results(training_generator, sample_size=500):
    legengine = leg_engine.LegEngine(max_simulations=70000)
    total_scanned = 0
    total_good_predictions = 0
    total_best_move_ranks = 0
    total_best_moves = 0
    total_hard_predictions = 0
    total_confident_good_predictions = 0
    average_confidence_while_winning_sum = 0
    average_confidence_while_winning_count = 0
    average_confidence_while_losing_sum = 0
    average_confidence_while_losing_count = 0
    numpy.set_printoptions(threshold=numpy.inf)
    for x in training_generator:
        encoded_board = x[0][0]
        one_record = x[1]
        pos_eval = category_to_eval(one_record[0][0])
        best_moves = one_record[1][0]
        all_best_moves = list(all_moves_from_encoded_array(best_moves).islice(0, 1))[0]
        simple_pos_eval = encoded_board_to_simple_eval(encoded_board)
        expanded_dims = numpy.expand_dims(encoded_board, 0)
        leg_predicted_eval, leg_encoded_top_moves = legengine.ai_model.predict(expanded_dims, verbose=0)
        leg_all_best_moves = trim_top_moves(
            list(all_moves_from_encoded_array(leg_encoded_top_moves[0]).islice(0, NUMBER_OF_TOP_MOVES_TO_INCLUDE)))
        confidence_of_winning = category_to_confidence_winning(leg_predicted_eval[0])
        if pos_eval < 0.5:
            average_confidence_while_losing_count += 1
            average_confidence_while_losing_sum += confidence_of_winning
        else:
            average_confidence_while_winning_count += 1
            average_confidence_while_winning_sum += confidence_of_winning
        simple_leg_predicted_eval = category_to_eval(leg_predicted_eval[0])
        simple_leg_predicted_eval_confidence = confidence_of_eval(leg_predicted_eval[0])
        prediction_diff = abs(pos_eval - simple_leg_predicted_eval)
        has_best_move = False
        best_move_rank = None
        current_rank = 0
        for next_move in leg_all_best_moves:
            current_rank += 1
            if all_best_moves[0].uci() == next_move[0].uci():
                has_best_move = True
                best_move_rank = current_rank
                break
        if abs(pos_eval - simple_pos_eval) > 0.02:
            total_hard_predictions += 1
        if has_best_move:
            total_best_moves += 1
            total_best_move_ranks += best_move_rank
        if prediction_diff < 0.01:
            total_good_predictions += 1
            if simple_leg_predicted_eval_confidence > 0.5:
                total_confident_good_predictions += 1
        total_scanned += 1
        best_move_percent = int((total_best_moves / total_scanned) * 100)
        if total_best_moves > 0:
            best_move_rank_avg = total_best_move_ranks / total_best_moves
        else:
            best_move_rank_avg = 0
        prediction_percent = int((total_good_predictions / total_scanned) * 100)
        if average_confidence_while_winning_count > 0:
            winner_conf_percent = int(
                (average_confidence_while_winning_sum / average_confidence_while_winning_count) * 100)
        else:
            winner_conf_percent = 0
        if average_confidence_while_losing_count > 0:
            loser_conf_percent = int(
                (average_confidence_while_losing_sum / average_confidence_while_losing_count) * 100)
        else:
            loser_conf_percent = 0
        if total_scanned >= sample_size:
            break
    print(
        f'{sample_size} samples. {TOP_MOVE_BEST_TO_WORST_SCORE_RATIO:.2f}/{NUMBER_OF_TOP_MOVES_TO_INCLUDE} (moves: {total_best_moves}/{total_scanned} {best_move_percent}% {best_move_rank_avg:.2f} avg rank'
        f' predictions: {total_good_predictions}/{total_scanned} {prediction_percent}%'
        f' wcp: {winner_conf_percent}% lcp: {loser_conf_percent}%'
        f' - difficult predictions: {total_hard_predictions} - confident good predictions: {total_confident_good_predictions})')


if __name__ == '__main__':
    scan_dateset_main()

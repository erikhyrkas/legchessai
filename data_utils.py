import random
import sys
import traceback
from functools import lru_cache

import chess
import numpy
from sortedcontainers import SortedList


# TRANSPOSE_DICT = {
#     'a': 'h',
#     'b': 'g',
#     'c': 'f',
#     'd': 'e',
#     'e': 'd',
#     'f': 'c',
#     'g': 'b',
#     'h': 'a'
# }


def invert_eval(predicted_eval):
    # current set for sigmoid/softmax (0 to 1)
    return 1 - predicted_eval
    # for tanh (-1 to 1)
    # return -1 * predicted_eval


# [PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING]
SIMPLE_PIECE_VALUES = [100, 300, 310, 500, 900, 10000]
SIMPLE_BEST_MOVE_GUESSES = ['d2d4', 'e2e4', 'b1c3', 'g1f3', 'f1e2', 'e1g1', 'c1e2', 'e1c1', 'b2c3', 'e2f3', 'g8f6',
                            'f8e7', 'c8f5', 'b1d7', 'e8g8', 'e8c8', 'f1g2', 'c1b2', 'c8b7', 'f8g7', 'f1e1', 'd4f3',
                            'h2h3', 'h7h6', 'c8g4', 'f2f4', 'c7c5', 'b7b5', 'f3d4', 'd1d4', 'c3e4', 'e5f4',
                            'f6d5', 'e7g5', 'c8f5', 'f6d5', 'g5f7', 'd1e2', 'e7g6', 'e5d4', 'b8c6', 'd7d6', 'f1b5',
                            'c1b2', 'c8b7', 'd5e4', 'b8d7', 'd8d4', 'd7d6', 'e7e6', 'a7a6', 'e4e5', 'd7d5', 'g1e2',
                            'd2f4', 'e4d2', 'f6e4', 'c7c6', 'b5e2', 'h7h5', 'h5h4']
MAX_SIMPLE_VALUE = (8 * SIMPLE_PIECE_VALUES[0]) + (2 * SIMPLE_PIECE_VALUES[1]) + (2 * SIMPLE_PIECE_VALUES[2]) + (
        2 * SIMPLE_PIECE_VALUES[3]) + SIMPLE_PIECE_VALUES[4] + SIMPLE_PIECE_VALUES[5]
EVAL_WINNING_THRESHOLD = 0.25  # .66 # 0.7
EVAL_OPPONENT_WINNING_THRESHOLD = 0.10  # invert_eval(EVAL_WINNING_THRESHOLD)
TOP_MOVE_CONFIDENCE_THRESHOLD = 0.001
# with 7 million position data set (TOP_MOVE_BEST_TO_WORST_SCORE_RATIO/NUMBER_OF_TOP_MOVES_TO_INCLUDE):
# 1.00/1 Wins 50.0% vs 1350
# 1.05/2 Wins 32.5% vs 1350
# 1.75/2 Wins 45.0% vs 1350
# 1.50/2 Wins 32.5% vs 1350
# 20.0/2 Wins 12.5% vs 1350
# 1.50/3 Wins 42.5% vs 1350
# 1.10/3 Wins 35.0% vs 1350
# 1.25/3 Wins 35.0% vs 1350
# 2.00/3 Wins 27.5% vs 1350
# 1.10/4 Wins 32.5% vs 1350
# 1.50/4 Wins 41.6% vs 1350
# 2.00/4 Wins  7.5% vs 1350

# 1.00/1 Wins 12.5% vs 1500
# 2.00/3 Wins 20.0% vs 1500
# 1.10/3 Wins 14.5% vs 1500
# 1.10/4 Wins 22.5% vs 1500

# 1.10/2 Wins  0.0% vs 1700
# 1.50/2 Wins  2.5% vs 1700
# 1.10/3 Wins  2.5% vs 1700
# 1.50/3 Wins 10.0% vs 1700
# 2.00/3 Wins  0.0% vs 1700
# 1.08/4 Wins  5.0% vs 1700
# 1.10/4 Wins  5.0% vs 1700
# 1.25/4 Wins  2.5% vs 1700
# 1.50/4 Wins  0.0% vs 1700
# 5.00/4 Wins  0.0% vs 1700
# 1.10/5 Wins  2.5% vs 1700
# -- with 10 million position data set (TOP_MOVE_BEST_TO_WORST_SCORE_RATIO/NUMBER_OF_TOP_MOVES_TO_INCLUDE):
# 1.000/1 Wins 52.5% vs 1400 (100 game sample: 52.5% wins)
# 1.020/2 Wins 49.0% vs 1400 (100 game sample: 49.0% wins)
# 1.025/2 Wins 54.5% vs 1400 (100 game sample: 54.5% wins)
# 1.040/2 Wins 41.5% vs 1400 (100 game sample: 41.5% wins)
# 1.045/2 Wins 45.5% vs 1400 (100 game sample: 45.5% wins)
# 1.049/2 Wins 37.0% vs 1400 (100 game sample: 37.0% wins)
# 1.050/2 Wins 55.5% vs 1400 (100 game sample: 55.5% wins)
# 1.053/2 Wins 47.5% vs 1400 (100 game sample: 47.5% wins)
# 1.055/2 Wins 49.0% vs 1400 (100 game sample: 49.0% wins)
# 1.060/2 Wins 46.0% vs 1400 (100 game sample: 46.0% wins)
# 1.075/2 Wins 46.0% vs 1400 (100 game sample: 46.0% wins)
# 1.100/2 Wins 49.5% vs 1400 (100 game sample: 49.5% wins)
# 1.050/3 Wins 48.5% vs 1400 (100 game sample: 48.5% wins)
# 1.000/1 Wins 41.5% vs 1450 (100 game sample: 41.5% wins)
# 1.025/2 Wins 42.5% vs 1450 (100 game sample: 42.5% wins)
# 1.000/1 Wins 27.0% vs 1500 (100 game sample: 27.0% wins)
# 1.000/1 Wins 19.5% vs 1600 (100 game sample: 19.5% wins)
# 1.000/1 Wins 20.0% vs 1700 (100 game sample: 16.0% wins)
# 1.001/2 Wins 12.5% vs 1700
# 1.002/2 Wins 15.0% vs 1700
# 1.003/2 Wins 17.5% vs 1700
# 1.004/2 Wins 10.0% vs 1700
# 1.005/2 Wins 20.0% vs 1700
# 1.006/2 Wins 12.5% vs 1700
# 1.008/2 Wins 20.0% vs 1700
# 1.050/2 Wins 10.0% vs 1700
# 1.100/2 Wins  5.0% vs 1700
# 1.500/2 Wins  5.0% vs 1700
# 100.0/2 Wins 10.0% vs 1700
# 1.040/3 Wins  2.5% vs 1700
# 1.045/3 Wins 20.0% vs 1700
# 1.050/3 Wins 22.5% vs 1700
# 1.051/3 Wins  5.0% vs 1700
# 1.055/3 Wins 20.0% vs 1700
# 1.060/3 Wins  0.0% vs 1700
# 1.100/3 Wins  5.0% vs 1700
# 1.500/3 Wins  5.0% vs 1700
# 100.0/3 Wins  5.0% vs 1700
# 1.040/4 Wins 10.0% vs 1700
# 1.045/4 Wins  5.0% vs 1700
# 1.049/4 Wins  10% vs 1700
# 1.050/4 Wins 16.3% vs 1700 (25.0/7.5)
# 1.051/4 Wins  5.0% vs 1700
# 1.055/4 Wins  7.5% vs 1700
# 1.060/4 Wins 10.0% vs 1700
# 1.070/4 Wins 15.0% vs 1700
# 1.080/4 Wins 17.5% vs 1700
# 1.100/4 Wins 17.5% vs 1700
# 1.120/4 Wins  5.0% vs 1700
# 1.550/4 Wins  2.5% vs 1700
# 100.0/4 Wins  0.0% vs 1700
# 1.011/5 Wins  5.0% vs 1700
# 1.050/5 Wins  5.0% vs 1700


TOP_MOVE_BEST_TO_WORST_SCORE_RATIO = 1.025  # 1.1  # 1.5  # 2  80=91%  90=91% 100 == 95%
NUMBER_OF_TOP_MOVES_TO_INCLUDE = 1  # 3  # 10  14 = 90% 15 == 95%
TOP_MOVES_DISTRIBUTION_CONSTANT = NUMBER_OF_TOP_MOVES_TO_INCLUDE
MAX_REBALANCE_RATE = 0.25
REBALANCES_PER_SEARCH = 1
CENTIPAWN_THRESHOLD_TO_KEEP_MOVES = 50
TIME_LIMIT_MARGIN = 0.015
BOARD_SHAPE: tuple[int, int, int] = (8, 8, 40)  # 40
MAX_CENTIPAWN_EVAL = 5000
SIMULATION_MISS_MAX = 20


def min_for_opponent(vals_array):
    len_vals_array = len(vals_array)
    if len_vals_array == 0:
        return 0
    if len_vals_array == 1:
        return vals_array[0]
    modulo = len_vals_array % 2
    min_val = None
    for index in range(0, len_vals_array - modulo, 2):
        if min_val is None:
            min_val = vals_array[index + 1]
        elif vals_array[index + 1] < min_val:
            min_val = vals_array[index + 1]
    return min_val


def average_by_min_of_pair(vals_array):
    len_vals_array = len(vals_array)
    if len_vals_array == 0:
        return 0
    if len_vals_array == 1:
        return vals_array[0]
    modulo = len_vals_array % 2
    val_sum = 0
    val_count = 0
    for index in range(0, len_vals_array - modulo, 2):
        val_sum += min(vals_array[index], vals_array[index + 1])
        val_count += 1
    return val_sum / val_count


def arithmetic_mean(vals_array):
    val_sum = 0
    for val in vals_array:
        val_sum += val
    return val_sum / len(vals_array)


def trimmed_mean(vals_array):
    if len(vals_array) < 4:
        return arithmetic_mean(vals_array)
    sorted_vals = SortedList(vals_array)
    sorted_vals.pop()
    sorted_vals.pop(0)
    return arithmetic_mean(sorted_vals)


def eval_to_category(predicted_eval_value):
    # category = round(predicted_eval_value * 4.0)
    result = numpy.zeros(3, dtype=numpy.int8)
    if predicted_eval_value == 0.5:
        result[1] = 1
    elif predicted_eval_value > 0.5:
        result[2] = 1
    else:
        result[0] = 1
    return result


def category_to_confidence_winning(category):
    # rationale: we want to take the move that is most winning
    # we value winning over not losing, but we really don't want to lose
    # sometimes there will be no winning move, so we have to take the best
    # not losing move.
    # by weighting winning with the "even" evaluation,
    # we attempt to avoid losing, while still preferring winning in most
    # situations.
    # Keep in mind that confidence is not always right
    # so, this also helps smooth out wrong confidences slightly.
    # winning confidence 1, draw confidence 0 = (1*3 + 0)/4 = 0.75
    # winning confidence 0.75, draw confidence 0 = (0.75*3 + 0)/4 = 0.56
    # winning confidence 0.5, draw confidence 0.25 = (0.5*3 + 0.25)/4 = 0.44
    # winning confidence 0.5, draw confidence 0.0 = (0.5*3 + 0.0)/4 = 0.38
    # winning confidence 0.25, draw confidence 0.75 = (0.25*3 + 0.75)/4 = 0.38
    # winning confidence 0.25, draw confidence 0.5 = (0.25*3 + 0.5)/4 = 0.31
    # winning confidence 0.25, draw confidence 0.25 = (0.25*3 + 0.25)/4 = 0.25
    # winning confidence 0, draw confidence 1 = (0*3 + 1)/4 = 0.25
    # winning confidence 0, draw confidence 0.75 = (0*3 + 0.75)/4 = 0.19
    #
    # winning_confidence = category[2] * 3
    # not_losing_confidence = 1.0 - category[0]
    # return (winning_confidence + not_losing_confidence) / 4
    # return (category[1]/2) + category[2]
    # return 1 - category[0]
    index = numpy.argmax(category)
    if index == 0:
        return 0.3
    if index == 1:
        return 0.5
    return 0.7
    # not_losing = 1 - category[0]
    # # return category[2]
    # if not_losing:
    #     return 1.0
    # return 0.0


def category_to_eval(category):
    index = numpy.argmax(category) + 1
    return index / 4.0


def confidence_of_eval(category):
    return category[numpy.argmax(category)]


def categorize_eval(predicted_eval_value):
    result = round(predicted_eval_value * 4.0) / 4.0
    return result


def is_eval_confidently_winning(predicted_eval):
    return predicted_eval >= EVAL_WINNING_THRESHOLD


def is_eval_confidently_losing(predicted_eval):
    return predicted_eval <= EVAL_OPPONENT_WINNING_THRESHOLD


def is_move_confidence_acceptable(move_confidence):
    return move_confidence > TOP_MOVE_CONFIDENCE_THRESHOLD


def scored_fen_to_board(scored_fen_string):
    scored_fen = parse_scored_fen(scored_fen_string)
    return chess.Board(scored_fen[0])


def encode_scored_fen(scored_fen_string):
    scored_fen = parse_scored_fen(scored_fen_string)
    encoded_fen, flipped = encode_fen(scored_fen[0])
    encoded_score = encode_score(scored_fen[1], flipped)
    return encoded_fen, encoded_score[0], encoded_score[1], flipped


def encode_fen(fen):
    current_board = chess.Board(fen)
    if not current_board.turn:
        current_board.apply_mirror()
        flipped = True
    else:
        flipped = False
    encoded_board = encode_board_for_white(current_board)
    return encoded_board, flipped


def encode_board(current_board):
    if not current_board.turn:
        return encode_board_for_white(current_board.mirror()), True
    return encode_board_for_white(current_board), False


def winning_eval(current_board, perspective_white):
    base_eval = simple_eval(current_board, perspective_white)
    # can't round because draw is 0.5
    if base_eval <= 0.5:
        return 0.0
    return 1.0


def simple_eval(current_board, perspective_white):
    # maybe change simple piece value of king to 0, since it would more greatly emphasize the value
    # of smaller pieces? The king is never absent on any valid board, even when mated.
    # this change would require the rebuild of the dataset after the change to SIMPLE_PIECE_VALUES[king]
    outcome = current_board.outcome()
    if outcome is not None:
        winner = outcome.winner
        if winner is None:
            return 0.5
        if current_board.turn == winner:
            return 1.0
        else:
            return 0.0
    white_score = 0.0
    black_score = 0.0
    for piece in chess.PIECE_TYPES:
        for _ in current_board.pieces(piece, chess.WHITE):
            white_score += SIMPLE_PIECE_VALUES[piece - 1]
        for _ in current_board.pieces(piece, chess.BLACK):
            black_score += SIMPLE_PIECE_VALUES[piece - 1]

    if perspective_white:
        diff = (white_score - black_score)
    else:
        diff = (black_score - white_score)
    result = ((diff / MAX_SIMPLE_VALUE) / 2.0) + 0.5
    return result


def simple_best_move_guess(current_board):
    dist_sum = 0
    top_moves_len = len(SIMPLE_BEST_MOVE_GUESSES)
    top_dist = []
    for i in range(top_moves_len):
        val = TOP_MOVES_DISTRIBUTION_CONSTANT + top_moves_len - i
        top_dist.append(val)
        dist_sum += val
    for i in range(top_moves_len):
        top_dist[i] = top_dist[i] / dist_sum

    result = []
    offset = 0
    for guess in SIMPLE_BEST_MOVE_GUESSES:
        move_guess = chess.Move.from_uci(guess)
        if current_board.is_legal(move_guess):
            result.append((move_guess, top_dist[offset]))
            offset += 1
    return result


def encode_board_for_white(current_board):
    if not current_board.turn:
        raise Exception("Should always be whites turn.")
    encoded_board = numpy.zeros(BOARD_SHAPE, dtype=numpy.float32)

    # layers 0 - 11 - piece positions
    for piece in chess.PIECE_TYPES:
        for square in current_board.pieces(piece, chess.WHITE):
            i, j = square_to_index(square)
            encoded_board[i][j][piece - 1] = 1
        for square in current_board.pieces(piece, chess.BLACK):
            i, j = square_to_index(square)
            encoded_board[i][j][piece + 5] = 1

    # layer 12 - en passant
    if current_board.has_legal_en_passant():
        i, j = square_to_index(current_board.ep_square)
        encoded_board[i][j][12] = 1
    # layer 13 - fifty move clock percent
    if current_board.halfmove_clock > 0:
        if current_board.halfmove_clock < 99:
            fifty_move_percent = 0
        else:
            fifty_move_percent = current_board.halfmove_clock / 100.0
        for i in range(8):
            for j in range(8):
                encoded_board[i][j][13] = fifty_move_percent
    # layer 14 - white king-side castling
    if current_board.has_kingside_castling_rights(chess.WHITE):
        for i in range(8):
            for j in range(8):
                encoded_board[i][j][14] = 1
    # layer 15 - white queen-side castling
    if current_board.has_queenside_castling_rights(chess.WHITE):
        for i in range(8):
            for j in range(8):
                encoded_board[i][j][15] = 1
    # layer 16 - black king-side castling
    if current_board.has_kingside_castling_rights(chess.BLACK):
        for i in range(8):
            for j in range(8):
                encoded_board[i][j][16] = 1
    # layer 17 - black queen-side castling
    if current_board.has_queenside_castling_rights(chess.BLACK):
        for i in range(8):
            for j in range(8):
                encoded_board[i][j][17] = 1

    for square in chess.SQUARES:
        i, j = square_to_index(square)
        # 10 possible attackers = 4 pawns (2 ep), 1 bishop, 2 knights,
        # 2 rooks, 1 queen, 1 king -- with 1 blocking
        white_attackers = current_board.attackers(chess.WHITE, square)
        # 18 - p, 19 - N, 20 - B, 21 - R, 22 - Q, 23 - K
        for white_attacker_square in white_attackers:
            piece = current_board.piece_at(white_attacker_square).piece_type
            encoded_board[i][j][piece + 17] = min(1, encoded_board[i][j][piece + 17] + 0.25)

        black_attackers = current_board.attackers(chess.BLACK, square)
        # 24 - p, 25 - N, 26 - B, 27 - R, 28 - Q, 29 - K
        for black_attacker_square in black_attackers:
            piece = current_board.piece_at(black_attacker_square).piece_type
            encoded_board[i][j][piece + 23] = min(1, encoded_board[i][j][piece + 23] + 0.25)

    for current_move in current_board.legal_moves:
        piece = current_board.piece_at(current_move.from_square).piece_type
        i, j = square_to_index(current_move.to_square)
        # 30 - p, 31 - N, 32 - B, 33 - R, 34 - Q, 35 - K
        encoded_board[i][j][piece + 29] = 1

    for current_move in current_board.legal_moves:
        i, j = square_to_index(current_move.from_square)
        encoded_board[i][j][36] = 1
        i, j = square_to_index(current_move.to_square)
        encoded_board[i][j][37] = 1

    for square in chess.SQUARES:
        i, j = square_to_index(square)
        white_attackers = len(current_board.attackers(chess.WHITE, square))
        black_attackers = len(current_board.attackers(chess.BLACK, square))
        encoded_board[i][j][38] = min(1, (white_attackers / 10.0))  # 10 = 4 pawns (2 ep), 1 bishop, 2 knights,
        encoded_board[i][j][39] = min(1, (black_attackers / 10.0))  # 2 rooks, 1 queen, 1 king -- with 1 blocking

    return encoded_board


def encoded_board_to_simple_eval(encoded_board):
    white_score = 0
    black_score = 0
    for piece in chess.PIECE_TYPES:
        for i in range(8):
            for j in range(8):
                if encoded_board[i][j][piece - 1] == 1:
                    white_score += SIMPLE_PIECE_VALUES[piece - 1]
                elif encoded_board[i][j][piece + 5] == 1:
                    black_score += SIMPLE_PIECE_VALUES[piece - 1]
    diff = white_score - black_score
    result = ((diff / MAX_SIMPLE_VALUE) / 2.0) + 0.5
    return result


def encoded_board_to_string(encoded_board):
    result_board = [['.' for _ in range(8)] for _ in range(8)]
    for piece in chess.PIECE_TYPES:
        for i in range(8):
            for j in range(8):
                if encoded_board[i][j][piece - 1] == 1:
                    result_board[i][j] = chess.PIECE_SYMBOLS[piece].upper()
                elif encoded_board[i][j][piece + 5] == 1:
                    result_board[i][j] = chess.PIECE_SYMBOLS[piece]

    result_string = ''
    # a1 = 0, 0 -> h8 = 7, 7
    for j in range(8):
        for i in range(8):
            result_string += f"{result_board[i][7 - j]} "
        result_string += '\n'

    return result_string


def transpose_move(uci):
    # print(f"transpose: {uci}")  # f4d6 -> became a7a6, but should have been f5d3
    first_letter = uci[0].lower()  # TRANSPOSE_DICT[uci[0].lower()]
    second_letter = 9 - int(uci[1])
    third_letter = uci[2].lower()  # TRANSPOSE_DICT[uci[2].lower()]
    fourth_letter = 9 - int(uci[3])
    if len(uci) > 4:
        fifth_letter = uci[4:]
    else:
        fifth_letter = ''
    result = f"{first_letter}{second_letter}{third_letter}{fourth_letter}{fifth_letter}"
    # print(result)
    return result


def encode_score(score, flipped):
    game_eval = 0.5
    if score[0] is not None:
        game_eval = centipawns_to_category_eval(score[0])
    elif score[1] is not None:
        if score[1] > 0:
            game_eval = 0.75
        elif score[1] == 0:
            game_eval = 0.0
        else:
            game_eval = 0.25
    if flipped:
        game_eval = invert_eval(game_eval)
    # print(f"black's turn, eval: {game_eval} {score}")
    # print(f"white's turn, eval: {game_eval} {score}")
    # non_mate_moves = SortedList(key=lambda x: -x[1])
    # mate_moves = SortedList(key=lambda x: x[1])
    # best_mate_score = 10
    # best_eval_score = -numpy.inf
    # for next_move in score[2]:
    #     if flipped:
    #         move_uci = transpose_move(next_move[0])
    #     else:
    #         move_uci = next_move[0]
    #     mate_score = next_move[2]
    #     if mate_score is not None:
    #         if mate_score > 0:
    #             mate_moves.add([move_uci, mate_score])
    #             if mate_score < best_mate_score:
    #                 best_mate_score = mate_score
    #     else:
    #         eval_score = next_move[1]
    #         non_mate_moves.add([move_uci, eval_score])
    #         if eval_score > best_eval_score:
    #             best_eval_score = eval_score
    #
    # kept_moves = []
    # if len(mate_moves) > 0:
    #     kept_moves.append(mate_moves[0][0])
    #     # for mate_move in mate_moves:
    #     #     if mate_move[1] == best_mate_score:
    #     #         kept_moves.append(mate_move[0])
    # elif len(non_mate_moves) > 0:
    #     kept_moves.append(non_mate_moves[0][0])
    #     # min_quality = int(best_eval_score - CENTIPAWN_THRESHOLD_TO_KEEP_MOVES)
    #     # for non_mate_move in non_mate_moves:
    #     #     if non_mate_move[1] >= min_quality:  # == best_eval_score:  #
    #     #         kept_moves.append(non_mate_move[0])
    kept_moves = []
    if len(score[2]) > 0:
        if flipped:
            move_uci = transpose_move(score[2][0][0])
        else:
            move_uci = score[2][0][0]
        kept_moves.append(move_uci)
    encoded_moves = encode_uci_moves(kept_moves)
    return eval_to_category(game_eval), encoded_moves


def simple_eval_to_category_eval(simple_val):
    if simple_val == 0.5:
        return 0.5
    if simple_val > 0.5:
        return 0.75
    return 0.25


def centipawns_to_category_eval(centipawns):
    # print(f"min({centipawns}, {MAX_CENTIPAWN_EVAL}) = {min(centipawns, MAX_CENTIPAWN_EVAL)} ")
    # print(f"max({min(centipawns, MAX_CENTIPAWN_EVAL)}, {-MAX_CENTIPAWN_EVAL}) = {max(min(centipawns, MAX_CENTIPAWN_EVAL), -MAX_CENTIPAWN_EVAL)}")
    # print(f"result / (2 * MAX_CENTIPAWN_EVAL)) + 0.5 = {(max(min(centipawns, MAX_CENTIPAWN_EVAL), -MAX_CENTIPAWN_EVAL) / (2 * MAX_CENTIPAWN_EVAL)) + 0.5}")
    if centipawns > 0:
        return 0.75
    elif centipawns < 0:
        return 0.25
    return 0.5
    # return (max(min(centipawns, MAX_CENTIPAWN_EVAL), -MAX_CENTIPAWN_EVAL) / (2 * MAX_CENTIPAWN_EVAL)) + 0.5


def encode_uci_moves(moves):
    label_length = get_uci_labels_length()
    result = numpy.zeros(label_length, dtype=numpy.int8)
    if len(moves) < 1:
        return result
    uci_encoder = get_uci_to_encoding_dict()
    for move in moves:
        offset = uci_encoder[move]
        result[offset] = 1
    return result


def moves_from_encoded_array(encoded_moves, board):
    result = SortedList(key=lambda x: -x[1])
    uci_to_encoded_moves = get_uci_to_encoding_dict()
    current_board = board
    for legal_move in current_board.legal_moves:
        legal_move_uci = legal_move.uci()
        if board.turn:
            as_white_legal_move_uci = legal_move_uci
        else:
            as_white_legal_move_uci = transpose_move(legal_move_uci)
        index = uci_to_encoded_moves[as_white_legal_move_uci]
        value = encoded_moves[index]
        if is_move_confidence_acceptable(value):
            result.add((legal_move, value))
    if len(result) > NUMBER_OF_TOP_MOVES_TO_INCLUDE:
        result = list(result.islice(0, NUMBER_OF_TOP_MOVES_TO_INCLUDE))
    return trim_top_moves(result)


def trim_top_moves(top_moves):
    if len(top_moves) < 2:
        return top_moves
    threshold = top_moves[0][1] / TOP_MOVE_BEST_TO_WORST_SCORE_RATIO
    result = SortedList(key=lambda x: -x[1])
    for move_score in top_moves:
        if move_score[1] < threshold:
            break
        result.add(move_score)
    return result


def all_moves_from_encoded_array(encoded_moves, turn=chess.WHITE):
    result = SortedList(key=lambda x: -x[1])
    uci_labels = get_uci_labels()
    for index in range(get_uci_labels_length()):
        if turn:
            legal_move_uci = uci_labels[index]
        else:
            legal_move_uci = transpose_move(uci_labels[index])
        legal_move = chess.Move.from_uci(legal_move_uci)
        value = encoded_moves[index]
        result.add((legal_move, value))
    return result


def parse_scored_fen(scored_fen):
    scored_fen_parts = scored_fen.strip().split(',')
    top_moves_parts = scored_fen_parts[3].strip().split(';')
    top_moves = parse_top_moves(top_moves_parts)
    if scored_fen_parts[1].strip() == '-':
        return scored_fen_parts[0].strip(), (None, int(scored_fen_parts[2].strip()), top_moves)
    return scored_fen_parts[0].strip(), (int(scored_fen_parts[1].strip()), None, top_moves)


def parse_top_moves(top_moves_parts):
    top_moves = []
    if top_moves_parts:
        for top_move_part in top_moves_parts:
            if len(top_move_part) > 0:
                top_move = tuple(top_move_part.split('/'))
                if top_move[1] == '-':
                    if top_move[2] == '-':
                        top_moves.append((top_move[0], None, 0))
                    else:
                        top_moves.append((top_move[0], None, int(top_move[2])))
                else:
                    top_moves.append((top_move[0], int(top_move[1]), None))
    return top_moves


def main_test():
    # print(transpose_move("f4d6"))  # f4d6 -> became a7a6, but should have been f5d3
    # print(transpose_move(transpose_move("f4d6")))
    # numpy.set_printoptions(threshold=numpy.inf)
    # for i in range(-15300, 15300, 100):
    #     print(f"{i} {encode_eval(i)}")
    # print(encode_scored_fen('8/7p/1pB3p1/8/P3pp2/6kP/3r4/6K1 w - - 2 40, -1699, -, g1f1/-2141/-;c6b5/-/-5;h3h4/-/-1;a4a5/-/-1;c6e4/-/-1'))
    #
    # test_board = chess.Board('8/7p/1pB3p1/8/P3pp2/6kP/3r4/6K1 w - - 2 40')
    # print(test_board)
    # print(encode_scored_fen(
    #     '8/7p/1pB3p1/8/P3pp2/6kP/3r4/6K1 w - - 2 40, -1699, -, g1f1/-2141/-;c6b5/-/-5;h3h4/-/-1;a4a5/-/-1;c6e4/-/-1')[
    #           1])
    # print(encode_scored_fen('8/7p/1p4p1/1B6/P3pp2/6kP/8/3r2K1 w - - 4 41, -, -3, b5f1/-/-3')[1])
    # print(encode_scored_fen('8/7p/1p4p1/P7/5p2/6kP/8/3r1q1K w - - 0 44, -, 0, ')[1])
    # test_board = chess.Board('8/7p/1p4p1/P7/5p2/6kP/8/3r1q1K w - - 0 44')
    # print(test_board)
    # print(test_board.outcome())
    # print()
    # # print(encode_scored_fen('8/7p/1p4p1/P7/5p2/6kP/8/3r1q1K w - - 0 44, -, 0, '))
    #
    print(encode_scored_fen(
        '4B1k1/5pbp/p5p1/8/QP2p3/P3P1P1/7P/5K2 b - - 0 29, 1123, -, g8h8/1177/-;g7e5/1279/-;f7f5/1286/-;f7f6/1322/-;a6a5/1585/-')[
              1])
    # print(encode_scored_fen(
    #     '4B1k1/3Q1pbp/p5p1/8/1P2p3/P3P1P1/7P/5K2 w - - 3 31, -, 3, e8f7/-/3;d7f7/-/8;b4b5/2146/-;f1e2/1471/-;a3a4/1294/-')[
    #           1])
    # print(encode_scored_fen('5Q1k/5B1p/p5p1/8/1P2p3/P3P1P1/7P/5K2 b - - 0 33, -, 0, ')[1])
    # test_board = chess.Board('5Q1k/5B1p/p5p1/8/1P2p3/P3P1P1/7P/5K2 b - - 0 33')
    # print(test_board)
    # print(test_board.outcome())
    # example_fen_1 = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1,0,-,d2d4/14/-;e2e4/0/-'
    # example_fen_3 = '4k2r/pb3p2/3q1PpR/2n1p1P1/npPrP3/3pQ3/PP1N1NB1/1R2K3 b k - 0 29,900,-,d7d5/14/-;h7h6/-10/-;b8c6/30/-'
    # test_board = chess.Board('4k2r/pb3p2/3q1PpR/2n1p1P1/npPrP3/3pQ3/PP1N1NB1/1R2K3 b k - 0 29')
    # next_encoded_fen, next_encoded_eval, next_encoded_best_moves, next_encoded_flipped = encode_scored_fen(
    #     example_fen_3)
    # print(test_board)
    # print(next_encoded_flipped)
    # print(encoded_board_to_string(next_encoded_fen))
    # example_fen_2 = 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1,0,-,d2d4/14/-;e2e4/-/1;h2h3/-10/-;a2a4/-/-1;f7f8q/-/2'
    # example_fen_4 = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1,-300,-,d2d4/14/-;e2e4/0/-'
    # example_fen_3 = 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1,900,-,d7d5/14/-;h7h6/-10/-;b8c6/30/-'
    # # print(parse_scored_fen(example_fen_1))
    # # print(encode_scored_fen(example_fen_4))
    # next_encoded_fen, next_encoded_eval, next_encoded_best_moves = encode_scored_fen(example_fen_3)
    # print(f"{next_encoded_fen}\n{next_encoded_eval}")
    # example_board_3 = chess.Board('rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1')
    # print(f"{moves_from_encoded_array(next_encoded_best_moves, example_board_3)}")
    # print(example_board_3)

    # print(get_uci_labels())
    # print(get_flipped_uci_labels())
    # print(get_uci_labels())
    # print(f"{get_uci_labels_length()}")
    # print(f"c2c4: {get_uci_to_encoding_dict()['c2c4']} vs 446: {get_uci_labels()[446]}")


@lru_cache()
def get_uci_labels_length():
    return int(len(get_uci_labels()))


@lru_cache()
def get_uci_labels():
    labels_array = []
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    numbers = ['1', '2', '3', '4', '5', '6', '7', '8']
    promoted_to = ['q', 'r', 'b', 'n']

    for file_index in range(8):
        for rank_index in range(8):
            destinations = [(destination_file_index, rank_index) for destination_file_index in range(8)] + \
                           [(file_index, destination_rank_index) for destination_rank_index in range(8)] + \
                           [(file_index + diagonal_index, rank_index + diagonal_index) for diagonal_index in
                            range(-7, 8)] + \
                           [(file_index + other_diagonal_index, rank_index - other_diagonal_index) for
                            other_diagonal_index in range(-7, 8)] + \
                           [(file_index + knight_file_index, rank_index + knight_rank_index) for
                            (knight_file_index, knight_rank_index) in
                            [(-2, -1), (-1, -2), (-2, 1), (1, -2), (2, -1), (-1, 2), (2, 1), (1, 2)]]
            for (destination_file_index, destination_rank_index) in destinations:
                if (file_index, rank_index) != \
                        (destination_file_index, destination_rank_index) and destination_file_index in range(8) \
                        and destination_rank_index in range(8):
                    move = letters[file_index] + numbers[rank_index] + letters[destination_file_index] + numbers[
                        destination_rank_index]
                    labels_array.append(move)
    for file_index in range(8):
        file_label = letters[file_index]
        for promoted_to_piece_label in promoted_to:
            labels_array.append(file_label + '2' + file_label + '1' + promoted_to_piece_label)
            labels_array.append(file_label + '7' + file_label + '8' + promoted_to_piece_label)
            if file_index > 0:
                left_diagonal_file_index = letters[file_index - 1]
                labels_array.append(file_label + '2' + left_diagonal_file_index + '1' + promoted_to_piece_label)
                labels_array.append(file_label + '7' + left_diagonal_file_index + '8' + promoted_to_piece_label)
            if file_index < 7:
                right_diagonal_file_index = letters[file_index + 1]
                labels_array.append(file_label + '2' + right_diagonal_file_index + '1' + promoted_to_piece_label)
                labels_array.append(file_label + '7' + right_diagonal_file_index + '8' + promoted_to_piece_label)
    return labels_array


@lru_cache()
def get_uci_to_encoding_dict():
    labels = get_uci_labels()
    label_count = get_uci_labels_length()
    label_mapping = {uci_move: index for uci_move, index in zip(labels, range(label_count))}
    return label_mapping


def square_to_index(square, log=False):
    i, j = chess.square_file(square), chess.square_rank(square)
    if log:
        printlog(f"square_to_index: {i}, {j} ({square}) {chess.square_name(square)}")
    return i, j
    # letter = chess.square_name(square)
    # if log:
    #     printlog(f"square_to_index: {8 - int(letter[1])}, {ord(letter[0]) - 97} ({square}) {letter}")
    # return 8 - int(letter[1]), ord(letter[0]) - 97


def index_to_square(i, j, log=False):
    result = chess.square(i, j)
    # letter = chess.square(j, 7 - i)  # letter-number
    if log:
        printlog(f"index_to_square: {j}, {i} ({result}) {chess.square_name(result)}")
    return result


def pick_random_from_distribution(dist, dist_sum):
    weighted_index = random.randint(0, dist_sum)
    current_index_start = 0
    found_index = None
    for index_weight in dist:
        if weighted_index <= index_weight:
            found_index = current_index_start
            break
        current_index_start += 1
    if found_index is None:
        found_index = 0
    return found_index


def printlog(val):
    print(val, file=sys.stderr)


if __name__ == '__main__':
    main_test()


def count_remaining_pieces(current_board):
    result = 0
    for piece in chess.PIECE_TYPES:
        for _ in current_board.pieces(piece, chess.WHITE):
            result += 1
        for _ in current_board.pieces(piece, chess.BLACK):
            result += 1
    return result


def find_end_game_move(current_board):
    end_move = None
    with chess.syzygy.open_tablebase("endgames/6-WDL") as tablebase:
        best_end_move_rating = -2
        for next_move in current_board.legal_moves:
            next_rating = tablebase.get_wdl(current_board)
            # next_rating = tablebase.probe_wdl(current_board) # only if we need to know what file is missing
            if next_rating is None:
                printlog(f"ERROR: MISSING MOVE {next_move} FROM END GAME TABLE!!!!")
                try:
                    tablebase.probe_wdl(current_board)
                except KeyError as err:
                    printlog(f"error details: {err}")
                    printlog(f"{traceback.format_exc()}")

            elif next_rating > best_end_move_rating:
                best_end_move_rating = next_rating
                end_move = next_move
    return end_move

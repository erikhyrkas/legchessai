import time
from functools import lru_cache

import chess
import chess.polyglot
import chess.syzygy
import keras.models as models
import numpy
import pathlib
from board_state_node import BoardStateTreeNode, move_paths_to_string
from data_utils import printlog, encode_fen, moves_from_encoded_array, category_to_confidence_winning, \
    count_remaining_pieces, find_end_game_move

DEFAULT_AI_MODEL = 'models/leg_model_2022_09_05_14_28_13.h5'


class LegEngine:
    # opening_book_file_path = "openingbooks/Human.bin"
    # opening_book_file_path = "openingbooks/baron30.bin"
    # opening_book_file_path = "openingbooks/Titans.bin"
    def __init__(self, opening_book_file_path="openingbooks/Titans.bin", use_endgame_tables=True, use_concurrency=False,
                 use_random_opening=False, ai_model_file=None, max_simulations=20, simulation_miss_max=20,
                 time_limit_margin=0.015):
        current_path = pathlib.Path(__file__).parent.resolve()
        if opening_book_file_path is not None:
            self.opening_book_file_path = f"{current_path}/{opening_book_file_path}"
        else:
            self.opening_book_file_path = None
        self.use_random_opening = use_random_opening
        self.use_endgame_tables = use_endgame_tables
        self.use_concurrency = use_concurrency
        self.max_simulations = max_simulations
        self.simulation_miss_max = simulation_miss_max
        self.time_limit_margin = time_limit_margin
        self.ai_model_file = ai_model_file or DEFAULT_AI_MODEL
        model_path = f"{current_path}/{self.ai_model_file}"
        self.ai_model = models.load_model(model_path)
        # self.ai_model.summary()

    @lru_cache(4096)
    def predict(self, fen):
        current_board = chess.Board(fen)

        if current_board.is_game_over():
            outcome = current_board.outcome()
            if outcome is not None:
                winner = outcome.winner
                if winner is None:
                    return 0.5, []  # 0.5 ... currently predicting confidence of winning
                if current_board.turn == winner:
                    return 1.0, []
                else:
                    return 0.0, []

        encoded_board, flipped = encode_fen(fen)
        expanded_dims = numpy.expand_dims(encoded_board, 0)
        predicted_eval_category, encoded_top_moves = self.ai_model.predict(expanded_dims, verbose=0)
        predicted_eval_value = category_to_confidence_winning(predicted_eval_category[0])
        top_moves = moves_from_encoded_array(encoded_top_moves[0], current_board)
        return predicted_eval_value, top_moves

    def find_best_move(self, fen, max_depth=1, time_limit=None):
        start_time = time.time()

        current_board = chess.Board(fen)
        if current_board.is_game_over():
            return None

        pieces_remaining = count_remaining_pieces(current_board)
        if (pieces_remaining > 31) and (self.opening_book_file_path is not None):
            if current_board.ply() == 0:
                # only look at the opening book if we're asked to use a random opening
                if self.use_random_opening:
                    with chess.polyglot.open_reader(self.opening_book_file_path) as opening_book:
                        opening_move = opening_book.choice(current_board)
                        if opening_move is not None:
                            # printlog(f"Book Move: {opening_move}")
                            return opening_move.move
            with chess.polyglot.open_reader(self.opening_book_file_path) as opening_book:
                opening_move = opening_book.get(current_board)
                if opening_move is not None:
                    # printlog(f"Book Move: {opening_move}")
                    return opening_move.move
                # printlog("Best opening move not found!")
        elif (pieces_remaining < 7) and self.use_endgame_tables:
            end_move = find_end_game_move(current_board)
            if end_move is not None:
                # printlog(f"End Game Move: {end_move}")
                return end_move
            # printlog(f"ERROR: MISSING END GAME TABLE DATA!!!!")

        best_move = self._best_move_search(fen, max_depth, start_time, time_limit)
        return best_move

    def _best_move_search(self, fen, max_depth, start_time, time_limit):
        board_state_tree = self._create_board_state_tree(fen)
        if time_limit is None:
            simulations_count = self.max_simulations
            simulation_miss_count = 0
            depth_first = True
            for _ in range(self.max_simulations):
                if not self.run_simulation(board_state_tree, max_depth, depth_first):
                    simulation_miss_count += 1
                else:
                    simulation_miss_count = 0
                if simulation_miss_count > self.simulation_miss_max:
                    if depth_first:
                        depth_first = False
                        simulation_miss_count = 0
                    else:
                        break
        else:
            simulations_count = 0
            cur_time = time.time()
            time_limit_with_margin = time_limit - self.time_limit_margin
            simulation_miss_count = 0
            depth_first = True
            while (cur_time - start_time) < time_limit_with_margin:
                if not self.run_simulation(board_state_tree, max_depth, depth_first):
                    simulation_miss_count += 1
                else:
                    simulation_miss_count = 0
                simulations_count += 1
                if simulation_miss_count > self.simulation_miss_max:
                    if depth_first:
                        depth_first = False
                        simulation_miss_count = 0
                    else:
                        break
                if simulations_count >= self.max_simulations:
                    break
                cur_time = time.time()

        move_paths = board_state_tree.get_best_move_paths(3)
        if len(move_paths) > 0:
            best_move_uci = move_paths[0][0]
            if best_move_uci is not None:
                best_move = chess.Move.from_uci(best_move_uci)
            else:
                best_move = None
        else:
            best_move_uci = None
            best_move = None
        # printlog(f"Predicted Move Paths:\n{move_paths_to_string(move_paths)}")
        # printlog(f"Simulation misses: {simulation_miss_count}")
        # printlog(f"Top moves from ai: {board_state_tree.top_moves}")
        # printlog(f"Best Move (after {simulations_count} simulations): {best_move_uci}")

        return best_move

    def run_simulation(self, board_state_tree, max_depth, depth_first):
        move, parent_node = board_state_tree.try_pick_unevaluated_move(0, max_depth, depth_first)
        if (move is not None) and (parent_node is not None):
            self._create_board_state_node(parent_node, move.uci())
            return True
        return False

    def _create_board_state_tree(self, fen):
        board_state_tree = self._cached_create_board_state_node(fen)
        return board_state_tree

    def _create_board_state_node(self, parent_node, move_uci):
        new_board = chess.Board(parent_node.fen)
        new_move = chess.Move.from_uci(move_uci)
        new_board.push(new_move)
        new_or_cached_node = self._cached_create_board_state_node(new_board.fen())
        parent_node.add_child(new_move.uci(), new_or_cached_node)

    @lru_cache(maxsize=4096)
    def _cached_create_board_state_node(self, fen):
        board = chess.Board(fen)
        predicted_eval, top_moves = self.predict(fen)
        return BoardStateTreeNode(board, predicted_eval, top_moves)

    def eval_for_current_player(self, current_board):
        # returns evaluation for current player
        move_eval, top_moves = self.predict(current_board.fen())
        # printlog(f"Eval: {move_eval:0.4f} {top_moves}")
        return move_eval

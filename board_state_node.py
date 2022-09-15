import random

import chess
import numpy.random
from sortedcontainers import SortedList

from data_utils import printlog, invert_eval, is_eval_confidently_winning, is_eval_confidently_losing, \
    pick_random_from_distribution, arithmetic_mean, TOP_MOVE_BEST_TO_WORST_SCORE_RATIO, trimmed_mean, \
    average_by_min_of_pair, min_for_opponent, EVAL_WINNING_THRESHOLD


def do_smoothing_strategy(base, child, depth):
    # return (base + child) / 2
    child_depth = depth + 1
    return (base + (child * depth)) / child_depth


def move_path_to_string(move_path):
    result = ''
    next_node = move_path
    delim = ''
    while next_node is not None:
        result += f'{delim}{next_node[0]} {next_node[1]:.4f} {next_node[2]:.4f} {next_node[3]:.4f} {next_node[4]:.4f} {next_node[5]:.4f}'
        delim = ' -> '
        next_node = next_node[6]
    return result


def move_paths_to_string(move_paths):
    result = 'move confidence_of_path position_confidence' \
             'move_confidence average_confidence confidence_for_maximizer\n'
    for move_path in move_paths:
        result += move_path_to_string(move_path) + '\n'
    return result


def move_path_to_brief_string(move_path):
    result = ''
    next_node = move_path
    delim = ''
    while next_node is not None:
        result += f'{delim}{next_node[0]} {next_node[1]:.4f} {next_node[2]:.4f} {next_node[3]:.4f} {next_node[4]:.4f} {next_node[5]:.4f}'
        delim = ' -> '
        next_node = next_node[6]
    return result


def move_paths_to_brief_string(move_paths):
    result = 'move confidence_of_path position_confidence' \
             'move_confidence average_confidence confidence_for_maximizer\n'
    for move_path in move_paths:
        result += move_path_to_string(move_path) + '\n'
    return result


class BoardStateTreeNode:
    def __init__(self, board: chess.Board, predicted_eval, top_moves):
        self.board = board
        self.fen = board.fen()
        self.top_moves = top_moves
        top_moves_len = len(self.top_moves)
        if top_moves_len > 0:
            dist_sum = 0
            top_dist = []
            for top_move in self.top_moves:
                val = int(top_move[1] * 1000)
                top_dist.append(val)
                dist_sum += val
            self.top_moves_distribution = top_dist
            self.top_moves_dist_sum = dist_sum
        else:
            self.top_moves_distribution = []
            self.top_moves_dist_sum = 0
        self.legal_moves = list(board.legal_moves)
        self.unexplored_legal_moves = list(board.legal_moves)
        self.top_moves_set = set()
        self.unexplored_top_moves = set()
        for top_move in top_moves:
            self.unexplored_top_moves.add(top_move[0].uci())
            self.top_moves_set.add(top_move[0].uci())
        self.predicted_eval = predicted_eval
        self.children = {}
        self.ordered_children = SortedList(key=lambda x: -x[1])

    def __repr__(self):
        lines = self.get_repr_core('', 0)
        return lines

    def get_repr_core(self, move, repr_depth):
        spaces = ''
        for _ in range(repr_depth):
            spaces = spaces + '-'
        lines = f"{spaces}{move} {self.predicted_eval} {self.fen} {self.top_moves}\n"
        for child in self.children:
            lines += self.children[child].get_repr_core(child, repr_depth + 1)
        return lines

    def __iter__(self):
        return self.children

    def add_child(self, move_uci, node):
        assert move_uci is not None
        assert node is not None
        assert isinstance(node, BoardStateTreeNode)
        self.ordered_children.add((move_uci, node.predicted_eval))
        self.children[move_uci] = node

    def get_child(self, move_uci):
        return self.children[move_uci]

    def try_pick_unevaluated_move(self, depth, max_depth=14, depth_first=True):
        if (max_depth is not None) and (depth >= max_depth):
            return None, None
        if depth_first:
            next_move = self.pick_random_move_depth_first(depth)
        else:
            next_move = self.pick_random_move_breadth_first()
        if next_move is None:
            return None, None
        next_move_uci = next_move.uci()
        if next_move_uci in self.children:
            child = self.children[next_move.uci()]
        else:
            child = None
        if child is None:
            # we found our target, an unexplored move
            return next_move, self
        return child.try_pick_unevaluated_move(depth + 1, max_depth)

    def pick_random_move_depth_first(self, depth):
        if len(self.top_moves) == 0:
            return None
        if depth > 0:
            # We only look at the top move for everything except for the top level children
            random_move = self.top_moves[0][0]
        else:
            random_move = self.pick_random_top_move()
        self.mark_move_explored(random_move)
        return random_move

    def pick_random_move_breadth_first(self):
        random_move = self.pick_unexplored_top_move()
        if random_move is None:
            random_move = self.pick_random_leading_move()
            if random_move is None:
                random_move = self.pick_random_top_move()
                if random_move is None:
                    # we can only get here if we have no top moves
                    random_move = self.pick_random_legal_move()
        self.mark_move_explored(random_move)
        return random_move

    def mark_move_explored(self, random_move):
        if random_move is not None:
            if random_move in self.unexplored_legal_moves:
                self.unexplored_legal_moves.remove(random_move)
            if random_move.uci() in self.unexplored_top_moves:
                self.unexplored_top_moves.remove(random_move.uci())

    def pick_random_top_move(self):
        if len(self.top_moves) == 0:
            return None
        if len(self.top_moves) == 1:
            return self.top_moves[0][0]
        found_index = pick_random_from_distribution(self.top_moves_distribution, self.top_moves_dist_sum)
        random_move = self.top_moves[found_index][0]
        return random_move

    def pick_random_leading_move(self):
        if len(self.ordered_children) == 0:
            return None
        if is_eval_confidently_losing(self.ordered_children[0][1]):
            return None
        if len(self.top_moves) == 0:
            # we don't have any top moves, so we're going to need to explore occasionally
            random_action = random.randint(1, 100)
            if random_action > 90:
                return None
        min_eval = self.ordered_children[0][1] / TOP_MOVE_BEST_TO_WORST_SCORE_RATIO
        max_range = 0
        for child in self.ordered_children:
            max_range += 1
            if child[1] < min_eval:
                break
        random_move_pair = random.choice(list(self.ordered_children.islice(0, max_range)))
        random_move = random_move_pair[0]
        return chess.Move.from_uci(random_move)

    def pick_random_legal_move(self):
        if len(self.legal_moves) == 0:
            return None
        if len(self.unexplored_legal_moves) > 0:
            return random.choice(list(self.unexplored_legal_moves))
        return random.choice(self.legal_moves)

    def pick_unexplored_top_move(self):
        if len(self.unexplored_top_moves) == 0:
            return None
        for move in self.top_moves:
            if move[0].uci() in self.unexplored_top_moves:
                return move[0]
        # all of our top moves are explored
        for move in self.top_moves:
            child = self.children[move[0].uci()]
            if len(child.unexplored_top_moves) > 0:
                return move[0]
        # all the children of our top moves are explored
        return None

    def get_best_path_nodes_for(self, move, maximizer, previous_confidence):
        child = self.children[move]
        position_confidence = child.predicted_eval
        move_confidence = self.get_move_confidence(move)
        adjusted_confidence = position_confidence
        # adjusted_confidence = position_confidence  # (position_confidence + move_confidence) / 2  # position_confidence  #
        # if position_confidence > 0.1:
        #     adjusted_confidence = 1.0
        # else:
        #     adjusted_confidence = 0.0
        if maximizer:
            confidence_for_maximizer = adjusted_confidence
        else:
            confidence_for_maximizer = invert_eval(adjusted_confidence)
        new_previous_confidence = previous_confidence.copy()
        new_previous_confidence.append(confidence_for_maximizer)

        # if maximizer:
        #     label = "me"
        # else:
        #     label = "you"
        if len(child.children) == 0:
            confidence_of_path = min(new_previous_confidence)  # new_previous_confidence[-1]  # min(new_previous_confidence)  # arithmetic_mean(new_previous_confidence)  # min(new_previous_confidence)  # arithmetic_mean(new_previous_confidence)
            # printlog(
            #     f"{label}\t\t{move}\t{move_confidence:0.6f}\t\t{position_confidence:0.6f}\t\t\t{adjusted_confidence:0.6f}\t\t\t{confidence_of_path:0.6f}\t{new_previous_confidence}")
            return (move, confidence_of_path, position_confidence,
                    move_confidence, adjusted_confidence, confidence_for_maximizer, None)
        # printlog(
        #     f"{label}\t\t{move}\t{move_confidence:0.6f}\t\t{position_confidence:0.6f}\t\t\t{adjusted_confidence:0.6f}\t\t\t????????\t{new_previous_confidence}")
        # the move with the best move confidence is likely the best move
        # however, we don't predict that it is winning, maybe we consider another move
        # Note: I don't know if the current evaluation is good.
        # we know the best move isn't always the one with the highest move confidence,
        # but it usually is. So, we know we only should pick a different move if we
        # are worried the top move is losing.
        result = SortedList(key=lambda x: -x[3])  # 3
        # child_max_confidence = 0
        # total_child_confidence = 0
        for child_move in child.children:
            next_child = child.get_best_path_nodes_for(child_move, not maximizer, new_previous_confidence)
            result.add(next_child)
            # child_max_confidence = max(next_child[1], child_max_confidence)
            # total_child_confidence += next_child[1]
        # Are any of our children winning?
        # If so, we'll have a higher standard for finding the right move.
        # Otherwise, always take the top move. This might be a bad plan.
        # average_child_confidence = total_child_confidence / len(child.children)
        result_node = None
        # if child_max_confidence > 0.2:
        for child_node in result:
            if child_node[1] >= 0.41:
                result_node = child_node
                break
        if result_node is None:
            result_node = result[0]
        confidence_of_path = result_node[1]
        return (move, confidence_of_path, position_confidence,
                move_confidence, adjusted_confidence, confidence_for_maximizer, result_node)

    def get_move_confidence(self, move_uci):
        if move_uci in self.top_moves_set:
            for move in self.top_moves:
                if move[0].uci() == move_uci:
                    return move[1]
        return 0

    def get_best_move_paths(self, limit=1):
        result = SortedList(key=lambda x: -x[1])
        for move in self.children:
            # printlog(f"get_best_move_paths: {move}")
            # printlog(
            #     "label\tmove\tmove_confidence\tposition_confidence\taverage_confidence\tconfidence_of_path\tnew_previous_confidence")
            result.add(self.get_best_path_nodes_for(move, True, []))
        # printlog(result)
        limited_result = list(result.islice(stop=limit))
        if len(limited_result) == 0:
            # (move, confidence_of_path, position_confidence,
            #                 move_confidence, average_confidence, confidence_for_maximizer, node)
            limited_result.append((self.top_moves[0][0].uci(), 0, 0, 0, 0, 0, None))
        # printlog(limited_result)
        return limited_result

    def get_best_move(self):
        paths = self.get_best_move_paths(1)
        if len(paths) == 0:
            return None
        first_path = paths[0]
        return first_path[0]

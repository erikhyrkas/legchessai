import pathlib
import sys
import chess.pgn
import os
from leg_engine import LegEngine, DEFAULT_AI_MODEL


def output_now(line):
    print(line)
    sys.stdout.flush()


def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    openings = None
    opening_book_file_path = pathlib.Path(__file__).parent.resolve()
    if opening_book_file_path is not None:
        opening_book_file_path = f"{opening_book_file_path}/openingbooks/Titans.bin"
        if os.path.exists(opening_book_file_path):
            openings = "openingbooks/Titans.bin"
    engine = LegEngine(opening_book_file_path=openings,
                       ai_model_file=DEFAULT_AI_MODEL,
                       max_simulations=70000, use_random_opening=False,
                       simulation_miss_max=20, use_endgame_tables=False)
    current_board = chess.Board()
    command_stack = []
    our_time = 1000
    while True:
        if command_stack:
            next_command = command_stack.pop()
        else:
            next_command = input()
        if next_command == 'quit':
            break
        elif next_command == 'uci':
            output_now('id name LEG Chess Engine 1.0')
            output_now('id author Erik, Logan, and Gavin Hyrkas')
            output_now('uciok')
        elif next_command == 'isready':
            output_now('readyok')
        elif next_command == 'ucinewgame':
            command_stack.append('position fen ' + chess.STARTING_FEN)
        elif next_command.startswith('position'):
            command_parts = next_command.split(' ')
            moves_index = next_command.find('moves')
            if moves_index >= 0:
                list_of_moves = next_command[moves_index:].split()[1:]
            else:
                list_of_moves = []
            if command_parts[1] == 'fen':
                if moves_index >= 0:
                    fen_parts = next_command[:moves_index]
                else:
                    fen_parts = next_command
                _, _, fen = fen_parts.split(' ', 2)
            elif command_parts[1] == 'startpos':
                fen = chess.STARTING_FEN
            else:
                pass
            current_board = chess.Board(fen)
            for next_move_text in list_of_moves:
                next_move = current_board.find_move(chess.parse_square(next_move_text[0:2]),
                                                    chess.parse_square(next_move_text[2:4]))
                current_board.push(next_move)
        elif next_command.startswith('go'):
            depth = 2
            move_time = None

            _, *params = next_command.split(' ')
            for param, val in zip(*2 * (iter(params),)):
                if param == 'depth':
                    depth = int(val)
                if param == 'movetime':
                    move_time = int(val)/1000.0
                if param == 'wtime':
                    our_time = int(val)/1000.0
            if our_time is not None:
                if move_time is None:
                    move_time = our_time
                elif our_time < move_time:
                    move_time = our_time
            best_move = engine.find_best_move(current_board.fen(), depth, move_time)
            if best_move is None:
                output_now('resign')
            elif current_board.is_legal(best_move):
                current_board.push(best_move)
            output_now(f"bestmove {best_move.uci()}")
        elif next_command.startswith('time'):
            our_time = int(next_command.split()[1])
        elif next_command.startswith('registration'):
            output_now('registration ok')
        else:
            pass


if __name__ == '__main__':
    main()

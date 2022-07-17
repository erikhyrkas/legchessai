from datetime import datetime
import sys

import chess.engine
import chess.pgn

from generate_pgn_from_engine import play

if __name__ == '__main__':
    engine_leg = chess.engine.SimpleEngine.popen_uci([sys.executable, r"leg_uci_engine.py"])
    engine_sf = chess.engine.SimpleEngine.popen_uci(r"stockfish\\stockfish_15_x64_avx2.exe")

    timestamp = datetime.now()
    for strength in range(1350, 2450, 50):
        engine_sf.configure({"UCI_LimitStrength": True})
        engine_sf.configure({"UCI_Elo": strength})
        play(timestamp, engine_leg, engine_sf, 1, "leg", f"stockfish_{strength}", None, 0.1, "exhibitions")

    for strength in range(1350, 2450, 50):
        engine_sf.configure({"UCI_LimitStrength": True})
        engine_sf.configure({"UCI_Elo": strength})
        play(timestamp,  engine_sf, engine_leg, 1, f"stockfish_{strength}", "leg", 0.1, None, "exhibitions")

    engine_leg.quit()
    engine_sf.quit()

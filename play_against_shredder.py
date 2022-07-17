from datetime import datetime
import sys

import chess.engine
import chess.pgn

from generate_pgn_from_engine import play

if __name__ == '__main__':
    engine_leg = chess.engine.SimpleEngine.popen_uci([sys.executable, r"leg_uci_engine.py"])
    engine_s = chess.engine.SimpleEngine.popen_uci(r"C:\\Program Files (x86)\\ShredderChess\\Deep Shredder 13\\EngineDeepShredder13UCIx64.exe")

    timestamp = datetime.now()
    for strength in range(850, 2450, 50):
        engine_s.configure({"UCI_LimitStrength": True})
        engine_s.configure({"UCI_Elo": strength})
        play(timestamp, engine_leg, engine_s, 1, "leg", f"shredder_{strength}", None, 0.1, "exhibitions")

    for strength in range(850, 2450, 50):
        engine_s.configure({"UCI_LimitStrength": True})
        engine_s.configure({"UCI_Elo": strength})
        play(timestamp,  engine_s, engine_leg, 1, f"shredder_{strength}", "leg", 0.1, None, "exhibitions")

    engine_leg.quit()
    engine_s.quit()

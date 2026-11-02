"""Backward-compatible command; implementation is in evaluation.replay_train."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from evaluation.replay_train import main

if __name__ == "__main__":
    result=main()
    if isinstance(result,int):raise SystemExit(result)

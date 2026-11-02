"""Backward-compatible command; implementation is in evaluation.run_text_baseline."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from evaluation.run_text_baseline import main

if __name__ == "__main__":
    result=main()
    if isinstance(result,int):raise SystemExit(result)

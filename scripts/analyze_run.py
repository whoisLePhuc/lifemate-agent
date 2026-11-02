from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[1]))

"""Frozen train/dev diagnostics; never import into agent inference."""
from evaluation.analysis import main

if __name__ == '__main__':
    main()

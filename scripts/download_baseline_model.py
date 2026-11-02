from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[1]))

"""Development-time download only; inference is strictly local_files_only."""
import argparse
from huggingface_hub import snapshot_download

p=argparse.ArgumentParser()
p.add_argument('--output-dir',required=True)
a=p.parse_args()
snapshot_download('Qwen/Qwen3-4B-Instruct-2507',
    revision='cdbee75f17c01a7cc42f958dc650907174af0554',local_dir=a.output_dir,
    allow_patterns=['*.json','*.safetensors','*.txt','*.jinja','README.md','LICENSE'],max_workers=3)

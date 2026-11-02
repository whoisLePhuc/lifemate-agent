from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[1]))

"""Train replay / test empty-session contract smoke, not an agent or scorer."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import re
import sys
import time

from lifemate.input.adapter import DatasetLoader
from lifemate.tools.env_adapter import LifecycleError, ToolEnvAdapter


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--split", choices=["train", "public_test"], required=True)
    p.add_argument("--limit", type=int)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.limit is not None and args.limit < 1: p.error("limit must be positive")
    if args.output.exists(): p.error("report already exists; choose a fresh run path")
    start = time.perf_counter()
    loader = DatasetLoader(args.data_root, args.split)
    count = calls = result_matches = hash_checks = validation_failures = ledger_matches = 0
    first_id = None
    data = loader.iter_training() if args.split == "train" else loader.iter_inputs()
    for row in data:
        if args.limit is not None and count >= args.limit: break
        item = row.agent_input if args.split == "train" else row
        first_id = first_id or item.id
        with ToolEnvAdapter(args.data_root, args.split, item.id) as env:
            results = []
            if args.split == "train":
                for call in row.reference.tool_calls:
                    valid = env.validate(call["name"], call["arguments"])
                    validation_failures += not valid["ok"]
                    results.append(env.call(call["name"], call["arguments"]))
                    calls += 1
            record = env.finalize()
            expected = row.reference.tool_calls if args.split == "train" else []
            ledger_matches += record["tool_calls"] == expected
            if args.split == "train":
                result_matches += results == row.reference.tool_results
            hash_checks += bool(re.fullmatch("[0-9a-f]{64}", record["state_hash"]))
        count += 1
    # This must reject before creating another official ToolEnv.
    duplicate_blocked = False
    try:
        ToolEnvAdapter(args.data_root, args.split, first_id)
    except LifecycleError:
        duplicate_blocked = True
    other_split = "public_test" if args.split == "train" else "train"
    mixed_split_blocked = False
    try:
        ToolEnvAdapter(args.data_root, other_split, "runtime_probe_not_a_sample")
    except LifecycleError:
        mixed_split_blocked = True
    passed = (count > 0 and ledger_matches == count and hash_checks == count
              and validation_failures == 0 and duplicate_blocked and mixed_split_blocked
              and (args.split != "train" or result_matches == count))
    report = {"utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
              "mode": "train_gold_replay" if args.split == "train" else "empty_session_smoke",
              "split": args.split, "samples": count, "executed_calls": calls,
              "ledger_matches": ledger_matches, "exact_tool_results_matches": result_matches if args.split == "train" else None,
              "valid_hash_format": hash_checks, "gold_validation_failures": validation_failures,
              "duplicate_sample_blocked": duplicate_blocked, "mixed_split_blocked": mixed_split_blocked,
              "final_state_match_checked": False, "score_checked": False,
              "passed": passed, "elapsed_seconds": round(time.perf_counter() - start, 3)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2); stream.write("\n")
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == "__main__": sys.exit(main())

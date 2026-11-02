from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[1]))

"""Replay train references, compare state, and write an explicitly gold submission."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import time

from lifemate.input.adapter import DatasetLoader
from evaluation.replay import TrainReplaySession
from lifemate.output.writer import SubmissionWriter, validate_submission


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.data_root.resolve(), args.output_dir.resolve()
    if out.is_relative_to(root):
        parser.error("outputs must be outside organiser Data")
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    loader = DatasetLoader(root, "train")
    ids = [item.id for item in loader.iter_inputs()]
    counters = {"samples": 0, "calls": 0, "call_ledger_matches": 0,
                "tool_results_matches": 0, "full_state_matches": 0,
                "label_apps": 0, "label_apps_matched": 0, "schema_rejections": 0}
    mismatches = []
    submission = out / "gold_replay_submission.jsonl"
    with SubmissionWriter(submission, ids) as writer:
        for example in loader.iter_training():
            item, ref = example.agent_input, example.reference
            with TrainReplaySession(root, item.id) as env:
                results = []
                for call in ref.tool_calls:
                    check = env.validate(call["name"], call["arguments"])
                    counters["schema_rejections"] += not check["ok"]
                    results.append(env.call(call["name"], call["arguments"]))
                    counters["calls"] += 1
                state = env.compare_final_state(ref.final_state)
                record = env.finalize()
            flags = {"call_ledger": record["tool_calls"] == ref.tool_calls,
                     "tool_results": results == ref.tool_results,
                     "full_state": state["full_state_match"]}
            counters["samples"] += 1
            for key in flags:
                counters[key + "_matches"] += flags[key]
            counters["label_apps"] += state["label_apps"]
            counters["label_apps_matched"] += state["label_apps_matched"]
            if not all(flags.values()):
                mismatches.append({"id": item.id, "checks": flags})
            writer.write(record)
    verified = validate_submission(submission, ids)
    passed = (bool(ids) and not mismatches and counters["schema_rejections"] == 0
              and counters["samples"] == len(ids))
    report = {"mode": "train_gold_replay_not_agent_predictions", "python": platform.python_version(),
              "utc": datetime.now(timezone.utc).isoformat(), "schema_version": loader.schema_version,
              "command": sys.argv, "data_root": str(root),
              **counters, "mismatches": mismatches, "submission_validation": verified,
              "passed": passed, "competition_score_computed": False,
              "elapsed_seconds": round(time.perf_counter() - start, 3)}
    with (out / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2); stream.write("\n")
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == "__main__": sys.exit(main())

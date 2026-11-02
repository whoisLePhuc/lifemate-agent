"""Validate metadata contracts; aggregate-only output, no labels or state dumps."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time

from lifemate_data import DatasetLoader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"python": platform.python_version(), "utc": datetime.now(timezone.utc).isoformat(),
              "data_root": str(args.data_root.resolve()), "splits": {}}
    for split in ["train", "public_test"]:
        start = time.perf_counter()
        loader = DatasetLoader(args.data_root, split)
        count = history = audio_refs = 0
        for item in loader.iter_inputs():
            count += 1
            history += bool(item.history)
            audio_refs += 1 + sum(turn["role"] == "user" for turn in item.history)
        references = sum(1 for _ in loader.iter_training()) if split == "train" else None
        report["splits"][split] = {
            "samples": count, "samples_with_history": history, "audio_references_checked": audio_refs,
            "training_references_checked": references, "schema_version": loader.schema_version,
            "catalog": loader.catalog_counts, "elapsed_seconds": round(time.perf_counter() - start, 3)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Never overwrite a previous run; choose a fresh run directory each time.
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

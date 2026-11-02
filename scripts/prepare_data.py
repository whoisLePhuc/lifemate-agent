"""Audit data and freeze grouped IDs; never use test labels or access app state."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import platform

from lifemate_data import DatasetLoader
from lifemate_data.preparation import audit_audio, distribution, group_train, split_groups


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True); stream.write("\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dev-fraction", type=float, default=.2)
    args = p.parse_args()
    if not 0 < args.dev_fraction < 1: p.error("invalid dev fraction")
    root, out = args.data_root.resolve(), args.output_dir.resolve()
    if out.is_relative_to(root): p.error("output must be outside Data")
    out.mkdir(parents=True, exist_ok=False)
    audit, train_digests = {}, None
    for split in ["train", "public_test"]:
        audit[split], digests = audit_audio(DatasetLoader(root, split))
        if split == "train": train_digests = digests
    save(out / "audio_audit.json", audit)
    if any(r["errors"] for r in audit.values()):
        raise ValueError("audio audit failed; see report; split not published")
    examples = list(DatasetLoader(root, "train").iter_training())
    groups, reasons = group_train(examples, train_digests)
    train, dev = split_groups(groups, dev_fraction=args.dev_fraction, seed=args.seed)
    train_set, dev_set = set(train), set(dev)
    crossed = sum(bool(set(g) & train_set) and bool(set(g) & dev_set) for g in groups)
    if crossed or train_set & dev_set or train_set | dev_set != {x.agent_input.id for x in examples}:
        raise AssertionError("split integrity failed")
    fingerprints = {name: sha256((root / name).read_bytes()).hexdigest() for name in [
        "VERSION", "train/train.jsonl", "train/train_tools_catalog.json"]}
    manifest = {"algorithm": "grouped-v1", "seed": args.seed, "dev_fraction_target": args.dev_fraction,
                "source_sha256": fingerprints, "train_ids": train, "dev_ids": dev,
                "groups": groups}
    # Reversed input order must produce the same groups and ID split.
    reverse_groups, _ = group_train(list(reversed(examples)), train_digests)
    if groups != reverse_groups or (train, dev) != split_groups(reverse_groups, dev_fraction=args.dev_fraction, seed=args.seed):
        raise AssertionError("order-independent reproducibility failed")
    save(out / "split_manifest.json", manifest)
    report = {"utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
              "groups": len(groups), "multi_sample_groups": sum(len(g)>1 for g in groups),
              "largest_group": max(map(len,groups)), "duplicate_evidence_counts": reasons,
              "cross_split_groups": crossed, "order_independent_reproducibility": True,
              "train": distribution(examples, train), "dev": distribution(examples, dev),
              "realized_dev_fraction": len(dev)/len(examples),
              "limitations": "Heuristic text/audio grouping, not verified conversation IDs or semantic deduplication; no stratification guarantee."}
    save(out / "split_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()

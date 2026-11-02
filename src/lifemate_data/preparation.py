"""Offline metadata/audio audit and deterministic grouped TRAIN split."""
from collections import Counter, defaultdict
from hashlib import sha256
import random
import re
import unicodedata
import wave


def normalize(text):
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFC", text).casefold()))


def audit_audio(loader):
    """Decode all referenced WAV frames; no environment access. Aggregate report only."""
    paths = set()
    for item in loader.iter_inputs():
        paths.add(item.audio)
        paths.update(t["audio"] for t in item.history if t["role"] == "user")
    formats, errors, durations, digests = Counter(), [], [], {}
    for path in sorted(paths):
        try:
            with wave.open(str(path), "rb") as stream:
                channels, rate, width = stream.getnchannels(), stream.getframerate(), stream.getsampwidth()
                count = stream.getnframes()
                if count <= 0 or rate <= 0:
                    raise ValueError("empty audio")
                digest = sha256(f"{channels}:{rate}:{width}:".encode())
                size = 0
                while True:
                    block = stream.readframes(65536)
                    if not block:
                        break
                    size += len(block); digest.update(block)
                if size != count * channels * width:
                    raise ValueError("truncated audio")
                formats[f"{channels}ch/{rate}Hz/{width * 8}bit"] += 1
                durations.append(count / rate)
                digests[path] = digest.hexdigest()
        except (OSError, EOFError, wave.Error, ValueError) as exc:
            errors.append({"audio": path.relative_to(loader.split_root).as_posix(),
                           "error_type": type(exc).__name__})
    return {"files": len(paths), "decoded": len(durations), "errors": errors,
            "formats": dict(formats), "total_hours": sum(durations) / 3600,
            "min_seconds": min(durations) if durations else None,
            "max_seconds": max(durations) if durations else None,
            "mean_seconds": sum(durations) / len(durations) if durations else None}, digests


def group_train(examples, audio_digests=None):
    """Conservative leakage heuristics, not a claim of semantic deduplication.

    Connect identical current text, identical complete user dialogue, shared
    substantial (>=6 token) user utterances in history/current, equal current
    decoded audio, and near-current text (>=8 tokens, trigram Jaccard >=.90).
    Names/numbers are kept. No gold actions are used to define groups.
    """
    ids = [x.agent_input.id for x in examples]
    if len(set(ids)) != len(ids) or len(ids) < 2:
        raise ValueError("need at least two unique train IDs")
    parent = list(range(len(ids)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    def union(i, j):
        i, j = find(i), find(j)
        if i != j: parent[max(i,j)] = min(i,j)
    owners, reasons, grams, index = {}, Counter(), [], defaultdict(set)
    for i, example in enumerate(examples):
        current = normalize(example.reference.audio_text)
        dialogue = [normalize(t) for t in example.reference.history_audio_text] + [current]
        keys = [("current_exact", current), ("dialogue_exact", tuple(dialogue))]
        keys += [("shared_user_utterance", t) for t in dialogue if len(t.split()) >= 6]
        if audio_digests is not None:
            keys.append(("current_audio_exact", audio_digests[example.agent_input.audio]))
        for key in keys:
            if key in owners:
                union(i, owners[key]); reasons[key[0]] += 1
            else: owners[key] = i
        tokens = current.split()
        g = set(zip(tokens, tokens[1:], tokens[2:])) if len(tokens) >= 8 else set()
        candidates = set().union(*(index[t] for t in g)) if g else set()
        for j in sorted(candidates):
            other = grams[j]
            if min(len(g), len(other)) / max(len(g), len(other)) < .90:
                continue
            if len(g & other) / len(g | other) >= .90:
                union(i, j); reasons["current_near_trigram"] += 1
        grams.append(g)
        for token in g: index[token].add(i)
    groups = defaultdict(list)
    for i, sample_id in enumerate(ids): groups[find(i)].append(sample_id)
    return sorted((sorted(g) for g in groups.values()), key=lambda g: g[0]), dict(reasons)


def split_groups(groups, *, dev_fraction=.2, seed=42):
    if not 0 < dev_fraction < 1:
        raise ValueError("dev_fraction must be between zero and one")
    flat = [i for g in groups for i in g]
    if any(not g for g in groups) or len(flat) != len(set(flat)) or len(groups) < 2:
        raise ValueError("need disjoint nonempty groups; cannot split a single connected group")
    ordered = sorted((sorted(g) for g in groups), key=lambda g: g[0])
    random.Random(seed).shuffle(ordered)
    target = max(1, round(len(flat) * dev_fraction))
    dev, train = [], []
    for group in ordered:
        if abs(len(dev) + len(group) - target) < abs(len(dev) - target): dev.extend(group)
        else: train.extend(group)
    if not dev or not train:
        raise ValueError("group constraints prevent nonempty split; review groups")
    return sorted(train), sorted(dev)


def distribution(examples, ids):
    wanted = set(ids)
    rows = [x for x in examples if x.agent_input.id in wanted]
    apps = Counter()
    for row in rows:
        apps.update({c["name"].split(".")[0] for c in row.reference.tool_calls})
    return {"samples": len(rows), "with_history": sum(bool(x.agent_input.history) for x in rows),
            "empty_gold": sum(not x.reference.tool_calls for x in rows),
            "ask_missing_info": sum(any(c["name"] == "sys.ask_missing_info" for c in x.reference.tool_calls) for x in rows),
            "gold_call_lengths": dict(sorted(Counter(len(x.reference.tool_calls) for x in rows).items())),
            "samples_by_gold_app": dict(sorted(apps.items()))}

"""Local LifeMate scorer. NOT an organiser-certified implementation.

Unspecified conventions are explicit: NFC/lower + whitespace tokenization,
multiset token F1, equal argument-key sets, ordered arrays, empty/empty F1=1.
Do not reuse the toolkit's accent-folding search helpers for scoring.
"""
from collections import Counter
import math
import unicodedata

SCORER_VERSION = 'lifemate-local-fuzzy-v1'
CONVENTIONS = {
    'certified_by_organisers': False,
    'normalization': 'Unicode NFC then lowercase; accents preserved',
    'tokenization': 'Python str.split (whitespace); punctuation preserved',
    'token_f1': 'multiset overlap; both empty=1; one empty=0',
    'fuzzy_threshold': 'strictly greater than 0.6',
    'argument_keys': 'identical sets; no implicit defaults',
    'arrays': 'ordered; recursive item schema',
    'matching': 'maximum cardinality one-to-one bipartite matching',
    'limitations': 'Tokenization, empty strings and nested arguments need organiser conformance.',
}


def token_f1(a, b):
    def tokens(value):
        return Counter(unicodedata.normalize('NFC', value).lower().split())
    aa, bb = tokens(a), tokens(b)
    na, nb = sum(aa.values()), sum(bb.values())
    if not na or not nb:
        return float(na == nb)
    return 2 * sum((aa & bb).values()) / (na + nb)


def exact_json(a, b):
    """JSON equality without Python's True == 1 type confusion."""
    if type(a) in (int, float) and type(b) in (int, float):
        return (type(a) is int or math.isfinite(a)) and (type(b) is int or math.isfinite(b)) and a == b
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(exact_json(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(exact_json(x, y) for x, y in zip(a, b))
    return a == b


def argument_match(a, b, schema):
    if schema.get('x-match') == 'fuzzy':
        # Fail closed for unsupported annotations rather than stringify IDs/objects.
        return isinstance(a, str) and isinstance(b, str) and token_f1(a, b) > .6
    if isinstance(a, dict) and isinstance(b, dict):
        props = schema.get('properties', {})
        return a.keys() == b.keys() and all(argument_match(a[k], b[k], props.get(k, {})) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        items = schema.get('items', {})
        return len(a) == len(b) and all(argument_match(x, y, items) for x, y in zip(a, b))
    return exact_json(a, b)


def match_calls(predicted, gold, tools, *, rejected_indices=()):
    schemas = {t['name']: t['parameters'] for t in tools}
    rejected = set(rejected_indices)
    if any(type(i) is not int or not 0 <= i < len(predicted) for i in rejected):
        raise ValueError('invalid rejected call index')
    edges = []
    for i, pred in enumerate(predicted):
        name = pred['name']
        edges.append([] if i in rejected or name not in schemas else [
            j for j, ref in enumerate(gold)
            if name == ref['name'] and argument_match(pred['arguments'], ref['arguments'], schemas[name])
        ])
    owners = {}

    def augment(i, visited):
        for j in edges[i]:
            if j in visited:
                continue
            visited.add(j)
            if j not in owners or augment(owners[j], visited):
                owners[j] = i
                return True
        return False

    for i in range(len(predicted)):
        augment(i, set())
    pairs = sorted((i, j) for j, i in owners.items())
    count = len(pairs)
    p = count / len(predicted) if predicted else 0.
    r = count / len(gold) if gold else 0.
    return {'pairs': pairs, 'matched_calls': count, 'predicted_calls': len(predicted),
            'gold_calls': len(gold), 'unmatched_predicted': len(predicted) - count,
            'precision': p, 'recall': r,
            'call_f1': 2 * count / (len(predicted) + len(gold)) if predicted or gold else 1.}


def score_sample(predicted, gold, tools, *, predicted_hash, gold_hash, rejected_indices=()):
    result = match_calls(predicted, gold, tools, rejected_indices=rejected_indices)
    state_match = predicted_hash == gold_hash
    # Explicit empty-gold rule is independent of the hash gate.
    score = float(not predicted) if not gold else result['call_f1'] if state_match else 0.
    return {**result, 'hash_match': state_match, 'local_score': score,
            'scorer_version': SCORER_VERSION, 'official_score': False}

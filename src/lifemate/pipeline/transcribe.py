"""Stage 1: audio -> text for the current request and every user history turn.

Reads only agent inputs (DatasetLoader.iter_inputs: no gold transcript, no label). There is no
fallback to reference text: an empty hypothesis stays empty and is counted. The default decoding
(beam 3, "guarded" temperature fallback) matches the PhoWhisper-medium LoRA evaluation. The guarded
profile can sample when it falls back, so decoding is not bit-reproducible; the per-audio fallback
temperature is recorded.

Needs faster-whisper/ctranslate2 (extra `asr`) and a CTranslate2 float16 model directory.
"""
import argparse, json, os, platform, sys, time
from pathlib import Path

from lifemate.input.adapter import DatasetLoader
from lifemate.observability.provenance import file_hash

PROFILES = {
    'greedy': {'temperature': 0.0},
    'guarded': {'temperature': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], 'compression_ratio_threshold': 2.4,
                'log_prob_threshold': -1.0},
}


def nvidia_library_path():
    import site
    base = Path(site.getsitepackages()[0]) / 'nvidia'
    return [str(base / 'cublas/lib'), str(base / 'cudnn/lib')]


def ensure_cuda_libraries():
    """CTranslate2 resolves cuBLAS/cuDNN through the loader path, which is fixed at process start."""
    wanted = nvidia_library_path()
    current = os.environ.get('LD_LIBRARY_PATH', '').split(':')
    if all(w in current for w in wanted):
        return
    env = dict(os.environ, LD_LIBRARY_PATH=':'.join(wanted + [c for c in current if c]))
    # Re-exec as a module so the same arguments work from `lifemate transcribe` and `python -m`.
    os.execve(sys.executable, [sys.executable, '-m', 'lifemate.pipeline.transcribe', *sys.argv[1:]], env)


def audio_paths(item):
    """Current request first, then user history turns in order."""
    paths = [Path(item.audio)]
    paths += [Path(t['audio']) for t in item.history if t['role'] == 'user']
    return paths


def assemble(sample_id, texts):
    if not texts:
        raise ValueError('at least the current request is required')
    return {'id': sample_id, 'request': texts[0], 'history': list(texts[1:])}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split', choices=['train', 'public_test', 'private_test'], required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True, help='CTranslate2 float16 ASR model directory')
    p.add_argument('--ids-file', type=Path, help='JSON list of sample IDs (required for train)')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--beam-size', type=int, default=3)
    p.add_argument('--decode', choices=list(PROFILES), default='guarded')
    a = p.parse_args()
    out = a.output_dir.resolve()
    if out.exists():
        raise SystemExit('choose a new output directory')
    if a.split == 'train' and not a.ids_file:
        raise SystemExit('train split requires --ids-file')
    ensure_cuda_libraries()
    import ctranslate2, faster_whisper
    from faster_whisper import WhisperModel

    wanted = None
    if a.ids_file:
        wanted = json.loads(a.ids_file.read_text())
        if not isinstance(wanted, list) or len(wanted) != len(set(wanted)) or not wanted:
            raise SystemExit('ids-file must be a nonempty list of unique IDs')
    data_root = a.data_root.resolve()
    items = [it for it in DatasetLoader(data_root, a.split).iter_inputs()
             if wanted is None or it.id in set(wanted)]
    if wanted is not None and {it.id for it in items} != set(wanted):
        raise SystemExit('some requested IDs are missing from the split')

    model_dir = a.model.resolve()
    model_hashes = {f.name: file_hash(f) for f in sorted(model_dir.iterdir()) if f.is_file()}
    out.mkdir(parents=True)
    started = time.perf_counter()
    model = WhisperModel(str(model_dir), device='cuda', compute_type='float16')
    load_seconds = time.perf_counter() - started
    infer = audio_total = 0.0
    empty = fallback = 0
    with (out / 'transcripts.jsonl').open('x', encoding='utf-8') as tx, \
            (out / 'audio.jsonl').open('x', encoding='utf-8') as au:
        for n, item in enumerate(items, 1):
            texts = []
            for index, path in enumerate(audio_paths(item)):
                t0 = time.perf_counter()
                segments, info = model.transcribe(
                    str(path), language='vi', task='transcribe', beam_size=a.beam_size,
                    condition_on_previous_text=False, vad_filter=False, without_timestamps=True,
                    **PROFILES[a.decode])
                segments = list(segments)
                text = ''.join(s.text for s in segments).strip()
                seconds = time.perf_counter() - t0
                temperature = max((s.temperature for s in segments), default=0.0)
                infer += seconds; audio_total += info.duration
                empty += not text; fallback += temperature > 0
                texts.append(text)
                au.write(json.dumps({'id': item.id, 'kind': 'request' if index == 0 else 'history',
                                     'index': index, 'path': str(path.relative_to(data_root)),
                                     'audio_sha256': file_hash(path), 'seconds': round(seconds, 4),
                                     'audio_seconds': round(info.duration, 4),
                                     'max_temperature': temperature, 'empty': not text},
                                    ensure_ascii=False) + '\n')
            tx.write(json.dumps(assemble(item.id, texts), ensure_ascii=False) + '\n')
            tx.flush(); au.flush()
            if n % 50 == 0 or n == len(items):
                print(f'{n}/{len(items)} samples', flush=True)
    meta = {'status': 'complete', 'split': a.split, 'samples': len(items),
            'audio_files': sum(len(audio_paths(i)) for i in items),
            'model': str(model_dir), 'model_files_sha256': model_hashes,
            'beam_size': a.beam_size, 'decode': a.decode, 'profile': PROFILES[a.decode],
            'language': 'vi', 'vad_filter': False, 'condition_on_previous_text': False,
            'load_seconds': round(load_seconds, 2), 'infer_seconds': round(infer, 2),
            'audio_seconds': round(audio_total, 2),
            'real_time_factor': round(infer / audio_total, 4) if audio_total else None,
            'empty_hypotheses': empty, 'fallback_decodes': fallback,
            'ids_file_sha256': file_hash(a.ids_file) if a.ids_file else None,
            'faster_whisper': faster_whisper.__version__, 'ctranslate2': ctranslate2.__version__,
            'python': platform.python_version(), 'gold_transcript_used': False}
    (out / 'meta.json').write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: v for k, v in meta.items() if k != 'model_files_sha256'}, indent=2))


if __name__ == '__main__':
    main()

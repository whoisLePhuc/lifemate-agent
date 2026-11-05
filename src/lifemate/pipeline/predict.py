"""One command: test-split audio -> ASR -> Qwen3.5 agent -> submission.jsonl -> organiser validator.

The two stages run in separate interpreters because their dependency stacks differ (CTranslate2 for
ASR, Torch/Transformers 5 for the LLM). Pass `--asr-python` and `--llm-python`, or leave both unset
to use the current interpreter when one environment has everything.

The submission is written to <output-dir>/agent/inference/submission.jsonl.
"""
import argparse, os, subprocess, sys
from pathlib import Path

ENV = {'USE_TF': '0', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'OMP_NUM_THREADS': '4',
       'TOKENIZERS_PARALLELISM': 'false'}


def build_commands(a, out):
    """The three commands, in order, as (argv, cwd). Pure so it can be tested without running anything."""
    asr_py, llm_py = str(a.asr_python), str(a.llm_python)
    ids = ['--ids-file', str(a.ids_file)] if a.ids_file else []
    transcribe = [asr_py, '-m', 'lifemate.cli', 'transcribe', '--split', a.split, '--data-root', str(a.data_root),
                  '--model', str(a.asr_model), '--beam-size', str(a.beam_size), '--output-dir', str(out / 'asr'), *ids]
    agent = [llm_py, '-u', '-m', 'lifemate.cli', 'run-agent', '--split', a.split, '--mode', 'asr',
             '--data-root', str(a.data_root), '--model-path', str(a.llm_model), '--adapter-path', str(a.adapter),
             '--transcripts', str(out / 'asr/transcripts.jsonl'), '--output-dir', str(out / 'agent'),
             '--seconds', str(a.seconds), *ids]
    validate = [sys.executable, '-m', 'zaic_tools', '--split', a.split, '--root', '..', '--validate',
                str(out / 'agent/inference/submission.jsonl')]
    return [(transcribe, None), (agent, None), (validate, Path(a.data_root) / a.split)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split', choices=['public_test', 'private_test'], required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--asr-model', type=Path, required=True, help='CTranslate2 float16 PhoWhisper directory')
    p.add_argument('--llm-model', type=Path, required=True, help='Qwen3.5-4B base model directory')
    p.add_argument('--adapter', type=Path, required=True, help='LoRA adapter directory')
    p.add_argument('--asr-python', type=Path, default=Path(sys.executable))
    p.add_argument('--llm-python', type=Path, default=Path(sys.executable))
    p.add_argument('--ids-file', type=Path, help='smoke subset; a real submission needs the full split')
    p.add_argument('--beam-size', type=int, default=3)
    p.add_argument('--seconds', type=float, default=30.0)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    out = a.output_dir.resolve()
    if out.exists():
        raise SystemExit('choose a new output directory')
    a.data_root = a.data_root.resolve()
    env = dict(os.environ, **ENV)
    for cmd, cwd in build_commands(a, out):
        print('+', ' '.join(cmd), flush=True)
        subprocess.run(cmd, cwd=cwd, env=env, check=True)
    print('submission:', out / 'agent/inference/submission.jsonl')


if __name__ == '__main__':
    main()

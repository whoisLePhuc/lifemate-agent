"""Unified source/installed CLI; imports only the selected workflow."""
import argparse
import importlib

COMMANDS = {
    'prepare-data': 'training.prepare_data',
    'replay-train': 'evaluation.replay_train',
    'text-baseline': 'evaluation.run_text_baseline',
    'analyze-run': 'evaluation.analysis',
    'replay-predictions': 'evaluation.replay_predictions',
    'evaluate-run': 'evaluation.evaluate_run',
    'compare-protocols': 'evaluation.compare_protocols',
    'compare-policies': 'evaluation.compare_policies',
    'prepare-trajectories': 'training.prepare_trajectories',
    'audit-trajectories': 'training.audit_trajectories',
    'train-sft': 'training.train_sft',
}

def main():
    import sys
    p=argparse.ArgumentParser(prog='lifemate',description='LifeMate offline workflows')
    p.add_argument('command',choices=COMMANDS)
    args=p.parse_args(sys.argv[1:2])
    remaining=sys.argv[2:]
    module=importlib.import_module(COMMANDS[args.command])
    sys.argv=[f'lifemate {args.command}',*remaining]
    result=module.main()
    return result if isinstance(result,int) else 0

if __name__ == '__main__':
    raise SystemExit(main())

"""Prepare a recorded scratch-training invocation; launches only with --run.

This is a launch/data-layout check, not a CUDA numerical preflight.
"""
import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['w64d8', 'w128d10', 'w128d20'], required=True)
    parser.add_argument('--protocol', choices=['detector', 'gt2d'], required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New, nonexistent directory')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    source = ROOT / f'configs/gcs_pose/{args.protocol}/{args.model}.yaml'
    config = yaml.safe_load(source.read_text(encoding='utf-8'))
    assert not config['finetune'] and not config['resume'] and not config['pretrained']
    data = args.data_root.resolve()
    metadata = data / config['dt_file']
    if not metadata.is_file():
        raise SystemExit(f'Missing metadata: {metadata}')
    counts = {}
    for split in ('train', 'test'):
        count = sum(len(list((data / subset / split).glob('*.pkl'))) for subset in config['subset_list'])
        if not count:
            raise SystemExit(f'Missing {split} clips under {data}')
        counts[split] = count
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config.update(data_root=str(data), seed=args.seed)
    effective = output / 'config.yaml'
    effective.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    command = [sys.executable, '-X', 'utf8', 'train.py', '--config', str(effective),
               '--checkpoint', str(output / 'checkpoints'), '--seed', str(args.seed)]
    record = dict(command=command, status='LAUNCH_REQUESTED' if args.run else 'PREPARED_ONLY',
                  config_sha256=hashlib.sha256(effective.read_bytes()).hexdigest(),
                  metadata_sha256=hashlib.sha256(metadata.read_bytes()).hexdigest(),
                  clips=counts, note='Layout checks only; no numerical preflight implied.')
    (output / 'launch.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(shlex.join(command))
    if args.run:
        result = subprocess.run(command, cwd=ROOT)
        record.update(status='PROCESS_EXITED', exit_code=result.returncode)
        (output / 'launch.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
        raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()

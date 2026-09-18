"""Bounded follow-up experiments; invoke explicitly on an authorized GPU."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def commands(data, output):
    base = [sys.executable, '-m', 'drone.grid_training.train_v2',
            '--data', data, '--steps', '50', '--batch', '32',
            '--epochs', '24', '--patience', '24']
    variants = [
        ('dino-finetuned', ['--arch', 'dino-small', '--pretrained', '--lr', '0.0003']),
        ('small-scratch', ['--arch', 'small', '--lr', '0.001']),
        ('resnet34-pretrained', ['--arch', 'resnet34', '--pretrained', '--lr', '0.001']),
        ('resnet18-seed2', ['--arch', 'resnet18', '--pretrained', '--lr', '0.001', '--seed', '1732']),
    ]
    return [base + ['--output', str(output / name)] + args for name, args in variants]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', default=os.environ.get('NORDIC_RUN_DIR'))
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not args.output:
        parser.error('--output or NORDIC_RUN_DIR is required')
    runs = commands(args.data, Path(args.output))
    print(json.dumps(runs, indent=2), flush=True)
    if args.dry_run:
        return
    for command in runs:
        subprocess.run(command, check=True)


if __name__ == '__main__':
    main()

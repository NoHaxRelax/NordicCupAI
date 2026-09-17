"""Verify the uploaded frozen data and fetch the exact official pretrained weights."""
import argparse
import json
from pathlib import Path
from ultralytics import YOLO
from train import sha, runtime, write

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    args=p.parse_args()
    data=args.root/'datasets/first-run-v2'
    manifest=json.loads((data/'manifest.json').read_text())
    for row in manifest['records']:
        assert sha(data/row['file']) == row['sha256'], row['file']
        if 'label_file' in row:
            assert sha(data/row['label_file']) == row['label_sha256'], row['label_file']
    path=args.root/'weights/yolo26x.pt'
    YOLO(str(path))
    expected='9fdd44a31c504547ffb81d2c6d9e6dac3493c8eaa8b0398d3f43bae6c7003e92'
    assert sha(path)==expected, 'Pretrained weights differ from HPC snapshot'
    write(args.root/'preflight.json', dict(runtime(), verified_images=len(manifest['records']),
          manifest_sha256=sha(data/'manifest.json'), pretrained_sha256=expected))
    print('PREFLIGHT_OK',len(manifest['records']))

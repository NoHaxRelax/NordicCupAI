"""A vertically flipped copy of the validation flight (first N frames): the ground then moves UP the frame, so the
auto band must choose the BOTTOM band, a branch no served run has exercised. Labels are flipped with the pixels.

    python build_flipped_scene.py --out "C:/.../drone-data/scenes/validation_vflip" --frames 120
"""
import argparse, json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import cv2
REPO = Path(__file__).resolve().parents[4]
SRC = REPO/'src'/'validation'
H = 2160


def one(args):
    n, out = args
    img = cv2.imread(str(SRC/'images'/f'frame_{n:06d}.png'), cv2.IMREAD_COLOR)
    cv2.imwrite(str(Path(out)/'images'/f'frame_{n:06d}.png'), cv2.flip(img, 0), [cv2.IMWRITE_PNG_COMPRESSION, 1])
    d = json.load(open(SRC/'annotations'/f'frame_{n:06d}.json'))
    for a in d['annotations']:
        x1, y1, x2, y2 = a['bbox']; a['bbox'] = [x1, H-y2, x2, H-y1]
    json.dump(d, open(Path(out)/'annotations'/f'frame_{n:06d}.json', 'w'))
    return n


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--frames', type=int, default=120)
    a = ap.parse_args()
    out = Path(a.out); (out/'images').mkdir(parents=True, exist_ok=True); (out/'annotations').mkdir(parents=True, exist_ok=True)
    json.dump({'source': 'validation flipped vertically', 'frames': a.frames}, open(out/'run_metadata.json', 'w'))
    with ProcessPoolExecutor(3) as ex:
        done = list(ex.map(one, [(n, str(out)) for n in range(1, a.frames+1)]))
    print('frames written', len(done))

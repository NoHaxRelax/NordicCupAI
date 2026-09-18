"""Save a motion model from two images; no service or submission is started."""
import argparse
import json
from pathlib import Path

import cv2

from .motion import ViewGeometry, calibrate_images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--first', type=Path, required=True)
    parser.add_argument('--second', type=Path, required=True)
    parser.add_argument('--first-request', type=Path, help='Saved request metadata for the first delivered crop')
    parser.add_argument('--second-request', type=Path, help='Saved request metadata for the second delivered crop')
    parser.add_argument('--source-size', type=int, nargs=2, default=(3840, 2160), metavar=('WIDTH', 'HEIGHT'))
    parser.add_argument('--first-tick', type=float, default=0.)
    parser.add_argument('--second-tick', type=float, default=1.)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    images, views = [], []
    for path, metadata in ((args.first, args.first_request), (args.second, args.second_request)):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            parser.error(f'Unable to read {path}')
        view = (ViewGeometry.from_request(json.loads(metadata.read_text())) if metadata else
                ViewGeometry(tuple(args.source_size), (0, 0, *args.source_size), image.shape[1::-1]))
        images.append(image); views.append(view)
    model = calibrate_images(*images, *views, first_tick=args.first_tick, second_tick=args.second_tick)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(model.to_dict(), indent=2, allow_nan=False)+'\n')
    print(json.dumps({'model': str(args.output), **model.diagnostics}, indent=2))


if __name__ == '__main__':
    main()

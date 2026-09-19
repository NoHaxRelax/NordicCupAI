"""Cut sprites of the validation objects the team never labelled and add them to elias/sprites/bank.json.

The objects come from elias/find_unlabelled2.py (chains of the Helsinki-only and the deployed detector); the three
ta-ta walkers were confirmed on the portal on 2026-09-19, the second small_tower and medium_launcher and the
small_launchers were judged against the Helsinki reference sheet. GrabCut runs with the bank's own settings
(sprites.cut_sprite, 3x enlargement, box growth allowed). Entries get track ids starting with "x-" and
label_status "agent_found", so they can be told apart or vetoed later.

    python elias/data/extra_sprites.py            # cuts, writes a review sheet, appends to the bank (backup kept)
    python elias/data/extra_sprites.py --dry      # cuts and writes the sheet only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ELIAS = HERE.parent
ROOT = ELIAS.parent
sys.path.insert(0, str(HERE))
from sprites import cut_sprite  # noqa: E402

# (search tag, chain index) -> class, from the contact sheets of 2026-09-19
WHAT = {('hel', 0): 'small_tower', ('both', 4): 'small_tower',
        ('hel', 3): 'medium_launcher', ('hel', 5): 'medium_launcher',
        ('hel', 8): 'ta-ta', ('hel', 9): 'ta-ta', ('hel', 17): 'ta-ta',
        }
# small_launcher chains (hel 2, 12; both 1, 7, 8) were tried: GrabCut returns empty or sliver masks on objects this
# small, and the labelled validation small_launchers already give nine sprites, so they are left out.
GROW = {'ta-ta': 0.35, 'small_launcher': 0.3}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--per-track', type=int, default=6); ap.add_argument('--dry', action='store_true')
    ap.add_argument('--classes', nargs='+', default=None, help='only these classes')
    ap.add_argument('--grow', type=float, default=None, help='box growth fraction for every class (default: per-class GROW, else 0.2)')
    ap.add_argument('--no-growth', action='store_true', help='forbid GrabCut from growing the box (keeps shadows out)')
    ap.add_argument('--suffix', default='', help='id suffix, e.g. -tight, so a re-cut does not collide with earlier entries')
    ap.add_argument('--sheet', default='extra_sprites_review.jpg')
    a = ap.parse_args()
    bank_path = ELIAS/'sprites'/'bank.json'; bank = json.loads(bank_path.read_text(encoding='utf-8'))
    classes = bank['classes']; known = {e['id'] for e in bank['sprites']}
    chains = {t: json.loads((ELIAS/'out'/f'unlabelled2_{t}.json').read_text()) for t in ('hel', 'both')}
    entries, sheet = [], []
    for (tag, k), cls in WHAT.items():
        if a.classes and cls not in a.classes:
            continue
        pts = sorted({p[0]: p for p in chains[tag][k]['pts']}.values())
        picks = [pts[int(i)] for i in np.linspace(0, len(pts)-1, min(a.per_track, len(pts)))]
        for p in picks:
            fr, x, y, w, h, conf = p[0], p[1], p[2], p[3], p[4], p[5]
            g = a.grow if a.grow is not None else GROW.get(cls, 0.2); bw, bh = w*(1+g), h*(1+g)
            box = [int(round(x-bw/2)), int(round(y-bh/2)), int(round(x+bw/2)), int(round(y+bh/2))]
            box = [max(4, box[0]), max(4, box[1]), min(3835, box[2]), min(2155, box[3])]
            img = cv2.imread(str(ROOT/'src'/'validation'/'images'/f'frame_{fr:06d}.png'), cv2.IMREAD_COLOR)
            sprite, origin, info = cut_sprite(img, box, not a.no_growth, 3)
            sid = f'cut:validation-f{fr:06d}-x-{cls}-{tag}{k}{a.suffix}'
            note = info.get('auto_reject', '')
            if sprite is None or note:
                print(f'  skip {sid}: {note or "no sprite"}'); continue
            sh, sw = sprite.shape[:2]; ox, oy = origin
            file = f'{cls}/cut__validation-f{fr:06d}-x-{cls}-{tag}{k}{a.suffix}.png'
            entry = {'file': file, 'id': sid, 'class_name': cls, 'class_id': classes.index(cls), 'zoom': 2, 'source': 'validation',
                     'frame': fr, 'track': f'x-{cls}-{tag}{k}{a.suffix}', 'label_status': 'agent_found', 'split': 'train',
                     'review_status': 'agent_found_2026-09-19_unlabelled_search', 'size': [int(sw), int(sh)],
                     'box_in_sprite': [box[0]-ox, box[1]-oy, box[2]-ox, box[3]-oy], 'box_src': box, 'origin_src': [int(ox), int(oy)],
                     'center_src': [float(x), float(y)], 'detector_confidence': float(conf), 'grabcut_variant': 'enlarged_3x',
                     'grabcut_box_grown': int(info.get('grown', 0)), 'fill_of_box': info.get('fill_of_box')}
            entries.append((entry, sprite))
            crop = img[max(0, box[1]-25):box[3]+25, max(0, box[0]-25):box[2]+25].copy()
            cv2.rectangle(crop, (box[0]-max(0, box[0]-25), box[1]-max(0, box[1]-25)), (box[2]-max(0, box[0]-25), box[3]-max(0, box[1]-25)), (0, 255, 255), 1)
            over = sprite[:, :, :3].copy(); over[sprite[:, :, 3] < 128] = (255, 0, 255)
            z = max(1, int(160/max(crop.shape[:2]))); crop = cv2.resize(crop, None, fx=z, fy=z, interpolation=cv2.INTER_NEAREST)
            over = cv2.resize(over, None, fx=z, fy=z, interpolation=cv2.INTER_NEAREST)
            tile = np.zeros((max(crop.shape[0], over.shape[0])+18, crop.shape[1]+over.shape[1]+6, 3), np.uint8)
            tile[18:18+crop.shape[0], :crop.shape[1]] = crop; tile[18:18+over.shape[0], crop.shape[1]+6:] = over
            cv2.putText(tile, f'{cls[:10]} f{fr} {tag}{k} {sw}x{sh} fill {info.get("fill_of_box")}', (2, 13), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 255, 255), 1)
            sheet.append(tile)
    print(f'{len(entries)} sprites cut')
    if sheet:
        w = max(t.shape[1] for t in sheet); h = max(t.shape[0] for t in sheet)
        pads = [cv2.copyMakeBorder(t, 0, h-t.shape[0], 0, w-t.shape[1], cv2.BORDER_CONSTANT) for t in sheet]
        pads += [np.zeros((h, w, 3), np.uint8)]*((-len(pads)) % 4)
        cv2.imwrite(str(ELIAS/'out'/a.sheet), np.vstack([np.hstack(pads[i:i+4]) for i in range(0, len(pads), 4)]))
        print('elias/out/'+a.sheet)
    if a.dry:
        return
    shutil.copy(bank_path, bank_path.with_suffix('.json.bak-20260919'))
    added = 0
    for entry, sprite in entries:
        if entry['id'] in known:
            continue
        out = ELIAS/'sprites'/entry['file']; out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), sprite)
        entry['sha256'] = hashlib.sha256(out.read_bytes()).hexdigest()
        bank['sprites'].append(entry); added += 1
    bank_path.write_text(json.dumps(bank, indent=1), encoding='utf-8')
    (ELIAS/'sprites'/'extra_entries_2026-09-19.json').write_text(json.dumps([e for e, _ in entries], indent=1), encoding='utf-8')
    print(f'{added} entries added to {bank_path.name}, backup at {bank_path.with_suffix(".json.bak-20260919").name}')


if __name__ == '__main__':
    main()

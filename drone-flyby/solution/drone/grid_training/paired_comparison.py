"""Compare saved crop predictions on the same fully contained objects."""
from collections import defaultdict
import json
from pathlib import Path


def main():
    data = Path('data/drone/grid-comparison-20260918-v1')
    root = Path('artifacts/drone-grid-training-20260918-v1')
    manifests = {s: json.loads((data/str(s)/'manifest.json').read_text()) for s in [256,384]}
    rows = {s: {r['id']: r for r in m['records'] if r['split']=='dev'} for s,m in manifests.items()}
    assert rows[256].keys() == rows[384].keys()
    def identity(a):
        return a['class_id'], a['group'], tuple(a['source_bbox_xyxy'])
    common = {}
    for ident in rows[256]:
        sets = [{identity(a) for a in rows[s][ident]['annotations'] if a['fully_contained']} for s in [256,384]]
        common[ident] = sets[0] & sets[1]
    outputs = {}
    for size in [256,384]:
        run = root/f'grid{size}-wave1-20260918'/f'resnet18-{size}-pretrained'
        preds = {r['id']:r for r in json.loads((run/'predictions.json').read_text())}
        groups = defaultdict(list)
        for ident, targets in common.items():
            p = preds[ident]
            top = max(range(16), key=lambda c:p['class_scores'][c])
            for c,group,box in targets:
                gate = p['object_score']>=.5 and p['class_scores'][c]>=.5
                groups[c,p['zoom'],group].append((float(gate and top==c),float(gate)))
        by_class_zoom = defaultdict(list)
        for (c,z,g), values in groups.items():
            by_class_zoom[c,z].append([sum(v[i] for v in values)/len(values) for i in [0,1]])
        per = {f'{manifests[size]["classes"][c]}/L{z}':
               {'top1_recall':sum(v[0] for v in vals)/len(vals),
                'multilabel_known_positive_recall':sum(v[1] for v in vals)/len(vals)}
               for (c,z),vals in by_class_zoom.items()}
        outputs[str(size)] = {'per_class_zoom':per,
                             'macro_top1_recall':sum(v['top1_recall'] for v in per.values())/len(per),
                             'macro_multilabel_recall':sum(v['multilabel_known_positive_recall'] for v in per.values())/len(per)}
    result = {'shared_object_views':sum(map(len,common.values())),
              'shared_tracks':len({(c,g) for targets in common.values() for c,g,b in targets}),
              'results':outputs,
              'limitations':'Same eligible object views, different context pixels and input resolutions. Selected on reused development data. Multi-label recall alone does not penalize wrong extra classes.'}
    (root/'paired-common-targets.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='results'},indent=2))
    print(json.dumps({s:{k:v for k,v in r.items() if k!='per_class_zoom'} for s,r in outputs.items()},indent=2))


if __name__ == '__main__':
    main()

"""Evaluate pixel refinement without retraining or changing raw reports."""
import argparse,json
from pathlib import Path
import cv2
from .refine import PixelRefiner
from .evaluate import measure
from .prepare import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,required=True);p.add_argument('--fixture',type=Path,required=True)
    p.add_argument('--bank',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();report=json.loads(a.report.read_text());refiner=PixelRefiner(a.bank)
    if report['zoom']!=2:raise ValueError('This diagnostic requires native-resolution proposals')
    fixture=json.loads((a.fixture/'fixture.json').read_text());lookup={(r['split'],r['frame']):r for r in fixture['examples']}
    for row in report['results']:
        source=lookup[(row['split'],row['frame'])];path=a.fixture/source['file']
        if sha(path)!=source['sha256']:raise ValueError('Image hash mismatch')
        row['predictions']=refiner.refine(cv2.imread(str(path)),row['predictions'])
        for threshold in row['thresholds']:
            rows=[p for p in row['predictions'] if p['score']>=float(threshold)]
            row['thresholds'][threshold]=dict(class_aware=measure(rows,row['truth']),class_agnostic=measure(rows,row['truth'],False))
    report.update(raw_report_sha256=sha(a.report),refinement_bank_sha256=sha(a.bank/'manifest.json'),refinement='small_launcher only; fixed bank-derived foreground palettes; unchanged class/confidence')
    with a.output.open('x') as f:json.dump(report,f,indent=2)


if __name__=='__main__':main()

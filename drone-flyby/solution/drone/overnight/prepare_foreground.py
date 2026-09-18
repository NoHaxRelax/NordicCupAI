"""Derive foreground-only detector labels without changing images or splits."""
import argparse,copy,os,tarfile
from collections import Counter
from pathlib import Path
from common import read,write,sha,fingerprint,verify_dataset

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--artifacts',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise SystemExit('Choose a new dataset version')
 original=read(a.source/'manifest.json');verify_dataset(a.source,original)
 m=copy.deepcopy(original);m['original_classes']=m['classes'];m['classes']=['object'];m['config']['classes']=['object']
 m['config']['experiments']=[dict(id='det-foreground-full',task='detector',initialization='pretrained',adaptation='full',zoom='mixed',epochs=100,lr=.0003),dict(id='det-foreground-partial',task='detector',initialization='pretrained',adaptation='partial',zoom='mixed',epochs=100,lr=.0005)]
 m['records']=[r for r in m['records'] if r['task']=='detector'];a.output.mkdir(parents=True)
 changed=[]
 for r in m['records']:
  target=a.output/r['file'];target.parent.mkdir(parents=True,exist_ok=True);os.link(a.source/r['file'],target)
  lines=(a.source/r['label_file']).read_text().splitlines();mapped=['0 '+line.split(' ',1)[1] for line in lines]
  # Coordinates and number of boxes are preserved exactly.
  assert [line.split()[1:] for line in lines]==[line.split()[1:] for line in mapped]
  label=a.output/r['label_file'];label.parent.mkdir(parents=True,exist_ok=True);label.write_text('\n'.join(mapped)+('\n' if mapped else ''))
  r['label_sha256']=sha(label);r['original_classes']=r['classes'];r['classes']=['object']*len(r['classes']);changed.append(r['label_file'])
 m['derivation']=dict(source_manifest_sha256=sha(a.source/'manifest.json'),source_fingerprint=original['source_fingerprint'],transform='class-id-to-zero; identical images, box coordinates, tracks and splits',code_sha256=sha(Path(__file__)))
 m['source_fingerprint']=fingerprint(dict(source=m['derivation'],config=m['config']))
 m['counts']=dict(Counter(r['task']+'/'+r['split'] for r in m['records']))
 m['limitations']+=['Foreground AP measures object localization only. Original 16-class labels must be used for end-to-end recognition evaluation.']
 write(a.output/'manifest.json',m);write(a.output/'config.json',m['config']);verify_dataset(a.output,m)
 a.artifacts.mkdir(exist_ok=True,parents=True)
 release=read(a.artifacts/'extra-views-release.json');release['source_fingerprint']=m['source_fingerprint'];release['annotation_evidence']+=' Foreground-only derivative changes class IDs to zero; preserves every image, box and split.';write(a.artifacts/'foreground-release.json',release)
 with tarfile.open(a.artifacts/'foreground-delta.tar.gz','w:gz') as t:
  for f in changed+['manifest.json','config.json']:t.add(a.output/f,arcname=f)
 write(a.artifacts/'foreground-integrity.json',dict(counts=m['counts'],source_fingerprint=m['source_fingerprint'],manifest_sha256=sha(a.output/'manifest.json'),delta_sha256=sha(a.artifacts/'foreground-delta.tar.gz'),dev_images_and_boxes_unchanged=True,records=len(m['records'])))
 print(m['counts'],m['source_fingerprint'])
if __name__=='__main__':main()

"""Reproduce the selected fixed-asset bank from frozen crops and corrected labels."""
import argparse,json,shutil,tempfile
from pathlib import Path
from .calibrate_poses import build as add_poses
from .detector import sha


def build(root,source,output):
 output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(dir=output.parent)as temporary:
  posed=Path(temporary)/'posed';add_poses(root,source,posed);shutil.copytree(posed,output)
 m=json.loads((output/'manifest.json').read_text());rows=[];audit=[]
 invalid={'validation-jet_plane-000043-38','validation-large_launcher-000060-18','validation-large_launcher-000077-35'}
 for r in m['templates']:
  if r['source']=='organizer_reference':rows.append(r);continue
  p=root/f"data/drone/training/score-anchored-validation-v7/{r['group']}.json";doc=json.loads(p.read_text());a=next((a for a in doc.get('annotations',[])if a['frame']==r['frame']),None)
  if r['id']in invalid or a is None or not(0<a['bbox_source_xyxy'][0]<a['bbox_source_xyxy'][2]<3839 and 0<a['bbox_source_xyxy'][1]<a['bbox_source_xyxy'][3]<2159):
   audit.append(dict(id=r['id'],action='exclude',reason='Visually invalid/background crop, no current annotation, or incomplete current box'));continue
  if a['class']!=r['class']:raise ValueError('Corrected class mismatch')
  x,y=r['bbox'][:2];b=a['bbox_source_xyxy'];r.update(annotation_extent=bool(r.get('calibration')),annotation_box=[b[0]-x,b[1]-y,b[2]-x,b[3]-y],annotation_file=str(p.relative_to(root)),annotation_sha256=sha(p),review_status=a['review_status']);rows.append(r);m['source_hashes'][str(p.relative_to(root))]=sha(p);audit.append(dict(id=r['id'],action='retain masked object pixels; calibrated crops project annotation extent, original crops project foreground',source_crop_bbox=r['bbox'],annotation_bbox=b))
 m['templates']=rows;m['classes']={c:sum(r['class']==c for r in rows)for c in m['classes']};m['validation_templates']=sum(r['source']!='organizer_reference'for r in rows);m['label_reconciliation']=dict(snapshot='score-anchored-validation-v7',new_validation_frames=0,audit=audit,policy='Matching uses masked object pixels. Original templates retain foreground geometry; six tight calibration crops project their corrected annotation extent.')
 (output/'manifest.json').write_text(json.dumps(m,indent=2));return m


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();m=build(a.root,a.source,a.output);print(json.dumps(dict(templates=len(m['templates']),validation_templates=m['validation_templates'])))

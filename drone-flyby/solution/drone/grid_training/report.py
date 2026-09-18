"""Render measured development failures from a completed checkpoint."""
import argparse,json
from pathlib import Path
from PIL import Image,ImageDraw

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--data',type=Path,required=True);a=p.parse_args();m=json.loads((a.data/'manifest.json').read_text());classes=m['classes'];errors=json.loads((a.run/'errors.json').read_text())
 errors.sort(key=lambda r: -r['object_score']*max(r['class_scores']) if r['kind']=='background' else r['class_scores'][r['missed_class']])
 unique={}
 for e in errors:unique.setdefault((e['id'],e.get('missed_class')),e)
 counts={};diverse=[]
 for e in unique.values():
  key=(e.get('missed_class','background'),e['zoom']);counts[key]=counts.get(key,0)+1
  if counts[key]<=2:diverse.append(e)
 errors=diverse[:48];sheet=Image.new('RGB',(4*260,((len(errors)+3)//4)*300 or 300),'#15202b');d=ImageDraw.Draw(sheet)
 for i,e in enumerate(errors):
  x=i%4*260;y=i//4*300;im=Image.open(a.data/e['file']);im.thumbnail((256,256));sheet.paste(im,(x,y+44));pred=max(range(16),key=lambda k:e['class_scores'][k]);true=classes[e['missed_class']] if 'missed_class' in e else 'background'
  d.text((x+3,y+3),f'True: {true} / L{e["zoom"]}',fill='white');d.text((x+3,y+20),f'Pred: {classes[pred]} {e["class_scores"][pred]:.2f}',fill='white')
 sheet.save(a.run/'failures.jpg');print(a.run/'failures.jpg')
if __name__=='__main__':main()

"""Generated geometric fixtures only. Never a competition accuracy dataset."""
import argparse
from pathlib import Path
from PIL import Image, ImageDraw
from common import read,sha,write


def make(output):
    config=read(Path(__file__).with_name('config.json'))
    output.mkdir(parents=True,exist_ok=False)
    records=[]
    for split in ['train','dev']:
        for z in range(3):
            for i in range(2):
                im=Image.new('RGB',(960,540),(35+i*20+(15 if split=='dev' else 0),70+z*15,80))
                x=150+i*220; y=120+z*35
                ImageDraw.Draw(im).rectangle([x,y,x+120,y+70],fill=(210,50,40))
                stem=f'{split}-L{z}-{i}'
                ip=f'detector/images/{split}/{stem}.png';lp=f'detector/labels/{split}/{stem}.txt'
                (output/ip).parent.mkdir(parents=True,exist_ok=True);im.save(output/ip)
                (output/lp).parent.mkdir(parents=True,exist_ok=True)
                (output/lp).write_text(f"15 {(x+60)/960} {(y+35)/540} {120/960} {70/540}\n")
                records.append(dict(task='detector',split=split,zoom=z,file=ip,sha256=sha(output/ip),label_file=lp,label_sha256=sha(output/lp)))
                for cls,box in [('tank',(x-8,y-8,x+128,y+78)),('background',(600,300,720,400))]:
                    cp=f'classifier/{split}/{cls}/{stem}.png';(output/cp).parent.mkdir(parents=True,exist_ok=True);im.crop(box).save(output/cp)
                    records.append(dict(task='classifier',split=split,zoom=z,file=cp,sha256=sha(output/cp),class_name=cls))
    write(output/'manifest.json',dict(schema=1,source_fingerprint='synthetic-smoke-only',classes=config['classes'],
        classifier_classes=config['classes']+['background'],config=config,records=records,counts={},coverage={},
        limitations=['Synthetic geometric smoke test. No competition data, quality claim, or usable checkpoint.']))
    write(output/'release.json',dict(annotations_frozen=True,validation_training_permitted=True,launch_enabled=True,
        source_fingerprint='synthetic-smoke-only',stop_starting_jobs_at='2026-09-18T06:00:00+00:00',
        permission_evidence='Self-generated rectangles solely for software verification; no validation footage.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);make(p.parse_args().output)

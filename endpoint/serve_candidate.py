"""Checkpoint 02: Elias released detector, resolution-aware tracking, selective L2."""
import os,sys
from pathlib import Path
ROOT=Path('/workspace/users/oscar/pipeline300')
settings=dict(DRONE_DETECTOR='ultralytics',DRONE_WEIGHTS=str(ROOT/'code/checkpoint02/both_m1280.pt'),
 DRONE_FAST_CALIBRATION='1',DRONE_DEVICE='cuda:0',DRONE_IMGSZ='1280',DRONE_HALF='1',DRONE_CONF='0.05',
 DRONE_BIRTH_CONFIDENCE='0.25',DRONE_UPDATE_CONFIDENCE='0.15',DRONE_CV_THREADS='4',
 DRONE_CONFIDENCE_MEMORY='8',DRONE_STRONG_CONFIDENCE='0.5',
 DRONE_OVERVIEW_BETWEEN_SIDES='0',DRONE_MISS_RULE='seen',DRONE_REVISIT_EVERY='8',
 DRONE_EXTENT_POLICY='blend',DRONE_EMIT_PARTIALS='1',DRONE_ENTRY_TRACKS='1',
 DRONE_LOG_DIR='/tmp/pipeline300-checkpoint02-frames',DRONE_ANSWER_WINDOWS='',DRONE_ANSWER_CLASSES='')
settings.update({k[5:]:v for k,v in os.environ.items() if k.startswith('CP02_DRONE_')})  # per-run overrides
os.environ.update(settings)
sys.path.insert(0,str(Path(__file__).parent))

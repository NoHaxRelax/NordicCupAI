"""Keep the endpoint answering on the serving pod: restart api.py when it stops.

Every 5 s GET /; every 60 s one real /predict with a grey warm-up frame (3 s timeout). Three failures in a row
restart api.py with the environment pod_serve.sh saved in /root/logs/serve.env. Runs on the pod:

    setsid nohup python elias/watchdog.py > /root/logs/watchdog.log 2>&1 &

The probe session is named 'watchdog' and costs one detector pass a minute.
"""
from __future__ import annotations

import base64
import os
import subprocess
import sys
import time

import requests

BASE = 'http://localhost:9053'
ENV = '/root/logs/serve.env'
LOG = '/root/logs/api.log'


def body(i):
    import cv2, numpy as np
    ok, png = cv2.imencode('.png', np.full((540, 960, 3), 90, np.uint8))
    return {'sequence_id': 'watchdog', 'frame': i, 'frame_index': i, 'request_id': f'watchdog:{i}', 'frame_interval_ms': 333,
            'response_timeout_ms': 3333, 'original_width': 3840, 'original_height': 2160, 'camera_command_feedback': None,
            'view': {'resolution_level': 0, 'center_x': 1920, 'center_y': 1080, 'view_id': f'w{i}', 'image': base64.b64encode(png.tobytes()).decode(),
                     'image_media_type': 'image/png', 'width': 960, 'height': 540, 'source_region_xyxy': [0, 0, 3840, 2160]},
            'camera_constraints': {'maximum_center_delta': 2203.0, 'allowed_resolution_levels': [0, 1], 'full_view_reset_exempt_from_delta': True,
                                   'center_bounds': [{'resolution_level': 0, 'width': 960, 'height': 540, 'minimum_center_x': 1920, 'maximum_center_x': 1920, 'minimum_center_y': 1080, 'maximum_center_y': 1080},
                                                     {'resolution_level': 1, 'width': 960, 'height': 540, 'minimum_center_x': 960, 'maximum_center_x': 2880, 'minimum_center_y': 540, 'maximum_center_y': 1620}]}}


def restart():
    print(time.strftime('%FT%T'), 'restarting api.py', flush=True)
    subprocess.call('pkill -f "[a]pi.py"; sleep 2', shell=True)
    # serve.env is the output of bash `export -p`: JSON values carry escaped quotes (DRONE_BOX_SCALE="{\"medium_launcher\": 0.85}"),
    # so let bash read it back instead of parsing it here (a naive strip of the outer quotes leaves the backslashes
    # and example.py then fails in json.loads at import).
    subprocess.Popen(['bash', '-c', f'. {ENV}; cd /root/work/drone-flyby && exec python api.py >> {LOG} 2>&1'], start_new_session=True)


def main():
    fails, i, last_predict = 0, 0, 0.
    while True:
        ok = False
        try:
            ok = requests.get(BASE+'/', timeout=5).status_code == 200
            if ok and time.time()-last_predict > 60:
                last_predict = time.time(); i += 1
                r = requests.post(BASE+'/predict', json=body(i), timeout=3); ok = r.status_code == 200
                print(time.strftime('%FT%T'), 'predict probe', r.status_code, f'{r.elapsed.total_seconds()*1000:.0f} ms', flush=True)
        except Exception as exc:
            print(time.strftime('%FT%T'), 'probe error', type(exc).__name__, str(exc)[:80], flush=True)
        fails = 0 if ok else fails+1
        if fails >= 3:
            restart(); fails = 0; time.sleep(30)
        time.sleep(5)


if __name__ == '__main__':
    sys.exit(main())

"""Per-process native adapter using event-streaming recorder v2 at 400px."""
from pathlib import Path
import hashlib
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / "debugger")]
import recorder as recorder_module
from streaming_recorder_v2 import ReplayRecorder as ReplayRecorderV2


class ReplayRecorder400(ReplayRecorderV2):
    def __init__(self, *args, **kwargs):
        kwargs["native_width"] = 400
        super().__init__(*args, **kwargs)


recorder_module.ReplayRecorder = ReplayRecorder400
import native_map_run_v2 as legacy
run = legacy.run
run.__globals__["OUT"] = ROOT / "results" / "short_overlap_sol" / "native_v4_stream"
SOURCE_HASHES = {
    **legacy.SOURCE_HASHES,
    "streaming_recorder_v2": hashlib.sha256(
        (ROOT / "debugger" / "streaming_recorder_v2.py").read_bytes()).hexdigest(),
    "stream_adapter_v4": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}


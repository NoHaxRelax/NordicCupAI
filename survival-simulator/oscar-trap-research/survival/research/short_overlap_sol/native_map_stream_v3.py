"""Resource-safe native-map adapter using the streaming replay recorder."""
from pathlib import Path
import hashlib
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / "debugger")]

# The frozen v2 adapter's dynamically compiled fixture imports `recorder`.
# Redirect only ReplayRecorder before importing it; all other recorder helpers
# and the v2 fixture/policy remain identical.
import recorder as recorder_module
from streaming_recorder import ReplayRecorder as StreamingReplayRecorder
recorder_module.ReplayRecorder = StreamingReplayRecorder

import native_map_run_v2 as legacy

run = legacy.run
run.__globals__["OUT"] = ROOT / "results" / "short_overlap_sol" / "native_v3_stream"
SOURCE_HASHES = {
    **legacy.SOURCE_HASHES,
    "streaming_recorder": hashlib.sha256(
        (ROOT / "debugger" / "streaming_recorder.py").read_bytes()).hexdigest(),
    "stream_adapter": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}


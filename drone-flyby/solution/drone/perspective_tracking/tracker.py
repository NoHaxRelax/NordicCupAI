"""One-observation object tracker using a frozen shared camera-motion model."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .motion import MotionModel, ViewGeometry, box_array, finite


OBJECT_CLASSES = frozenset(("hangar", "helicopter", "jet_plane", "large_launcher", "large_tower",
    "medium_launcher", "medium_plane", "mine_roller", "small_launcher", "small_plane", "small_tower",
    "ta-ta", "tank", "condor", "jammer", "spacecraft"))


@dataclass(frozen=True)
class Track:
    track_id: str
    label: str
    box: tuple[float, float, float, float]
    tick: float
    confidence: float


class DeterministicTracker:
    """One instance per sequence; instance IDs are distinct from class labels.

    add() supplies one complete box. Predictions always warp its original
    corners, so querying intermediate frames cannot accumulate box growth.
    refresh() is optional and explicit; predict() never changes an anchor.
    """
    def __init__(self, model: MotionModel, sequence_id: str):
        if not isinstance(sequence_id, str) or not sequence_id:
            raise ValueError("A nonempty sequence ID is required")
        self.model = model
        self.sequence_id = sequence_id
        self._tracks: dict[str, Track] = {}

    @property
    def tracks(self):
        return tuple(self._tracks.values())

    def _observation(self, track_id, label, box, tick, confidence):
        if not isinstance(track_id, str) or not track_id:
            raise ValueError("A nonempty instance track ID is required")
        if label not in OBJECT_CLASSES:
            raise ValueError(f"Unknown object class: {label}")
        box = box_array(box)
        if np.any(box[:2] < 0) or np.any(box[2:] > self.model.source_size):
            raise ValueError("Initial box must lie within the source frame")
        tick, confidence = map(float, finite([tick, confidence], "observation"))
        if tick < self.model.origin_tick:
            raise ValueError("Object observation predates the calibration's time origin")
        if not 0 <= confidence <= 1:
            raise ValueError("Confidence must be between zero and one")
        self.model.mapping(tick, tick)
        return Track(track_id, label, tuple(float(v) for v in box), tick, confidence)

    def add(self, track_id, label, box, tick, *, confidence=1.):
        if track_id in self._tracks:
            raise ValueError("Track already exists; use refresh explicitly")
        self._tracks[track_id] = self._observation(track_id, label, box, tick, confidence)

    def add_view_detection(self, track_id, label, box, tick, view: ViewGeometry,
                           *, normalized=False, confidence=1.):
        if view.source_size != self.model.source_size:
            raise ValueError("View source dimensions do not match the calibrated model")
        self.add(track_id, label, view.box_to_source(box, normalized=normalized), tick,
                 confidence=confidence)

    def refresh(self, track_id, box, tick, *, confidence=None):
        old = self._tracks[track_id]
        if tick <= old.tick:
            raise ValueError("Refresh must be newer than the previous observation")
        self._tracks[track_id] = self._observation(track_id, old.label, box, tick,
                                                  old.confidence if confidence is None else confidence)

    def remove(self, track_id):
        del self._tracks[track_id]

    def predict(self, track_id, tick):
        """Return source/global boxes, or None when entirely outside the image.

        A camera crop does not limit predictions: source-frame boxes remain
        available after the camera pans elsewhere. A projection outside the
        model's valid time domain raises ProjectionError, rather than wrapping.
        """
        track = self._tracks[track_id]
        tick = float(finite(tick, "tick"))
        if tick < self.model.calibrated_until:
            raise ValueError("Prediction must wait until shared calibration is available")
        if tick < track.tick:
            raise ValueError("Cannot predict before this track's latest observation")
        full = self.model.box(track.box, track.tick, tick)
        clipped = np.clip(full, 0, np.tile(self.model.source_size, 2))
        if np.any(clipped[2:] <= clipped[:2]):
            return None
        return {"track_id": track.track_id, "object_id": track.label, "tick": tick,
                "bbox_source_xyxy": clipped.tolist(), "bbox_unclipped_xyxy": full.tolist(),
                "bbox": (clipped/np.tile(self.model.source_size, 2)).tolist(),
                "confidence": track.confidence, "anchor_tick": track.tick}

    def predictions(self, tick):
        result = []
        for track in self.tracks:
            prediction = self.predict(track.track_id, tick)
            if prediction is not None:
                result.append(prediction)
        return result

    def response(self, request, *, tick=None, requested_view=None):
        """Return the organizer response schema, without networking or detection.

        Default time is frame_index, so skipped requests advance the full gap.
        Pass an explicit motion tick when using a separately measured clock.
        The actual delivered crop is irrelevant to existing track predictions.
        """
        if request["sequence_id"] != self.sequence_id:
            raise ValueError("Sequence changed; create a fresh tracker and calibration")
        if (request["original_width"], request["original_height"]) != self.model.source_size:
            raise ValueError("Request source dimensions do not match calibration")
        rows = self.predictions(request["frame_index"] if tick is None else tick)
        if len(rows) > 500 or any(sum(r["object_id"] == label for r in rows) > 100 for label in OBJECT_CLASSES):
            raise ValueError("Predictions exceed organizer annotation limits")
        return {"request_id": request["request_id"], "frame": request["frame"],
                "annotations": [{k: row[k] for k in ("object_id", "bbox", "confidence")} for row in rows],
                "requested_view": requested_view}

    def to_dict(self):
        return {"version": 1, "sequence_id": self.sequence_id, "model": self.model.to_dict(),
                "tracks": [{"track_id": t.track_id, "label": t.label, "box": list(t.box),
                            "tick": t.tick, "confidence": t.confidence} for t in self.tracks]}

    @classmethod
    def from_dict(cls, data):
        if data.get("version") != 1:
            raise ValueError("Unsupported tracker state version")
        tracker = cls(MotionModel.from_dict(data["model"]), data["sequence_id"])
        for track in data["tracks"]:
            tracker.add(track["track_id"], track["label"], track["box"], track["tick"],
                        confidence=track["confidence"])
        return tracker

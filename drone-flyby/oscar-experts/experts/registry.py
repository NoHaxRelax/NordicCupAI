"""One place that knows how to build the expert for any class."""
from .hangar import HangarExpert
from .condor import CondorExpert
from .generic import GenericExpert
from .specs import SPECS
from .blob import BlobExpert, BlobSpec

BLOBS = {'small_launcher': BlobSpec('small_launcher', probability=.25, use_lightness=False, size_window=(.5, 2.5)),
         'medium_launcher': BlobSpec('medium_launcher', probability=.2, use_lightness=True, size_window=(.5, 2.2))}

FAMILIES = {'hangar': ('hangar_expert', 'sift'), 'condor': ('condor_expert', 'pixel', 'sift')}
CLASSES = ['condor', 'hangar', 'helicopter', 'jammer', 'jet_plane', 'large_launcher', 'large_tower', 'medium_launcher',
           'medium_plane', 'mine_roller', 'small_launcher', 'small_plane', 'small_tower', 'spacecraft', 'ta-ta', 'tank']


def load_gates(path):
    import json
    from pathlib import Path
    return json.loads(Path(path).read_text())['gates'] if path else {}


class Gated:
    """Post-filter wrapper for experts that do not apply a gate themselves (hangar, condor).
    Candidate features are untouched; rows below the gate are marked rejected_by='gate'."""
    def __init__(self, expert, gate, family):
        self.expert, self.gate, self.family = expert, gate, family
        self.class_name, self.settings = expert.class_name, expert.settings

    def __getattr__(self, name):
        return getattr(self.expert, name)

    def detect(self, image, pixels_per_source_pixel=1., zoom=None, explain=False, **kwargs):
        import numpy as np
        from .fit_gates import apply_gate
        try:
            out = self.expert.detect(image, pixels_per_source_pixel, zoom, explain=True, **kwargs)
        except TypeError:
            out = self.expert.detect(image, pixels_per_source_pixel, zoom, explain=True)
        if isinstance(out[0], dict):
            families, candidates = out
        else:
            accepted, sift_rows, candidates = out
            families = {self.family: accepted, 'sift': sift_rows}
        z = int(zoom if zoom is not None else 2)
        kept = []
        for row in families[self.family]:
            logit, passed = apply_gate(self.gate, row, z)
            row.update(gate_logit=float(logit), gate_probability=float(1 / (1 + np.exp(-logit))), gate_pass=bool(passed))
            row['score'] = row['gate_probability']
            if passed:
                kept.append(row)
            else:
                row['rejected_by'] = 'gate'
        families[self.family] = kept
        return (families, candidates) if explain else families


def make_expert(class_name, bank, gates=None):
    gate = (gates or {}).get(class_name)
    if class_name == 'hangar':
        expert = HangarExpert(bank)
        return Gated(expert, gate, 'hangar_expert') if gate else expert
    if class_name == 'condor':
        expert = CondorExpert(bank)
        return Gated(expert, gate, 'condor_expert') if gate else expert
    if class_name in BLOBS:
        return BlobExpert(bank, BLOBS[class_name], gate=gate)
    if class_name not in SPECS:
        raise ValueError(f'No expert spec for {class_name}')
    return GenericExpert(bank, SPECS[class_name], gate=gate)


def families(class_name):
    return FAMILIES.get(class_name, ('expert', 'sift'))

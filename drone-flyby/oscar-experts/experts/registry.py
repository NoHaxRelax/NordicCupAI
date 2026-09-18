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


def make_expert(class_name, bank, gates=None):
    gate = (gates or {}).get(class_name)
    if class_name == 'hangar':
        return HangarExpert(bank)
    if class_name == 'condor':
        return CondorExpert(bank)
    if class_name in BLOBS:
        return BlobExpert(bank, BLOBS[class_name], gate=gate)
    if class_name not in SPECS:
        raise ValueError(f'No expert spec for {class_name}')
    return GenericExpert(bank, SPECS[class_name], gate=gate)


def families(class_name):
    return FAMILIES.get(class_name, ('expert', 'sift'))

"""Load a frozen fixed-asset detector with verified model and template hashes."""
import json
from pathlib import Path
from .fixed_assets import FixedAssetDetector, FixedAssetSettings
from .prepare import sha


def load_bundle(manifest, device='cpu'):
    manifest = Path(manifest).resolve()
    doc = json.loads(manifest.read_text())
    if doc.get('format') not in ('fixed-asset-detector-v1','fixed-asset-detector-v2'):
        raise ValueError('Unsupported detector bundle')
    paths = {'weights':None,'heatmap':None}
    names=['weights','verifier','heatmap','bank'] if doc['format']=='fixed-asset-detector-v1' else ['verifier','bank']+[n for n in ['weights','heatmap'] if n in doc['files']]
    for name in names:
        entry = doc['files'][name]
        path = (manifest.parent / entry['path']).resolve()
        target = path/'manifest.json' if name == 'bank' else path
        if sha(target) != entry['sha256']:
            raise ValueError('Bundle checksum mismatch: ' + name)
        paths[name] = path
    settings = {**doc['settings'], 'device': device}
    settings['heatmap_classes'] = tuple(settings['heatmap_classes'])
    if doc['format']=='fixed-asset-detector-v2':
        if settings.get('cnn_enabled',True) and paths['weights'] is None:raise ValueError('Bundle enables CNN proposals without weights')
        if settings.get('heatmap_enabled',True) and paths['heatmap'] is None:raise ValueError('Bundle enables heatmap proposals without weights')
    return FixedAssetDetector(paths['weights'], paths['bank'],
                              FixedAssetSettings(**settings), paths['verifier'],
                              paths['heatmap'])

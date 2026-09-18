"""Run a trained scratch CNN on one delivered camera image."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import cv2
from .detector import ScratchDetector,CNNSettings
from .prepare import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--weights',type=Path)
    source.add_argument('--bundle',type=Path)
    p.add_argument('--image',type=Path,required=True)
    p.add_argument('--source-region',type=float,nargs=4,required=True)
    p.add_argument('--threshold',type=float);p.add_argument('--device',default='cpu')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--checkpoint-sha256')
    p.add_argument('--feature-bank',type=Path);p.add_argument('--verifier',type=Path)
    p.add_argument('--feature-upsample',type=float,default=2.)
    p.add_argument('--feature-ransac-iterations',type=int,default=3000)
    p.add_argument('--fixed-assets',action='store_true')
    p.add_argument('--heatmap-weights',type=Path)
    a=p.parse_args()
    import torch
    torch.set_num_threads(4);cv2.setNumThreads(4)
    image=cv2.imread(str(a.image),cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim!=3 or image.shape[2] not in [3,4]:raise ValueError('Invalid input image')
    if image.shape[2]==4 and (image[:,:,3]!=255).any():raise ValueError('Unobserved pixels in delivered image')
    if a.bundle and (a.feature_bank or a.verifier or a.fixed_assets or a.heatmap_weights or a.checkpoint_sha256 or a.threshold is not None):
        raise ValueError('Bundle mode specifies its own checkpoints, bank and settings')
    if bool(a.feature_bank)!=bool(a.verifier):raise ValueError('Hybrid mode requires both --feature-bank and --verifier')
    if a.checkpoint_sha256 and sha(a.weights)!=a.checkpoint_sha256:raise ValueError('Checkpoint hash mismatch')
    if a.bundle:
        from .bundle import load_bundle
        detector=load_bundle(a.bundle,a.device);settings=detector.settings
    elif a.fixed_assets:
        if not a.feature_bank or not a.verifier or not a.heatmap_weights:
            raise ValueError('Fixed-asset mode requires --feature-bank, --verifier and --heatmap-weights')
        from .fixed_assets import FixedAssetDetector,FixedAssetSettings
        settings=FixedAssetSettings(cnn_threshold=.6 if a.threshold is None else a.threshold,device=a.device)
        detector=FixedAssetDetector(a.weights,a.feature_bank,settings,a.verifier,a.heatmap_weights)
    elif a.feature_bank:
        from .hybrid import HybridDetector,HybridSettings
        settings=HybridSettings(cnn_threshold=.25 if a.threshold is None else a.threshold,device=a.device,feature_upsample=a.feature_upsample,feature_ransac_iterations=a.feature_ransac_iterations)
        detector=HybridDetector(a.weights,a.feature_bank,a.verifier,settings)
    else:
        settings=CNNSettings(score_threshold=.25 if a.threshold is None else a.threshold,device=a.device)
        detector=ScratchDetector(a.weights,settings,a.checkpoint_sha256)
    start=time.perf_counter();predictions=detector.predict(image[:,:,:3],a.source_region)
    result=dict(predictions=predictions,seconds=time.perf_counter()-start,settings=asdict(settings),source_region=a.source_region,
                checkpoint_sha256=detector.checkpoint_sha256,image_sha256=sha(a.image))
    if a.feature_bank or a.bundle:result.update(feature_bank_sha256=detector.bank_sha256,verifier_sha256=detector.verifier.checkpoint_sha256)
    if (a.fixed_assets or a.bundle) and detector.heatmap is not None:result['heatmap_sha256']=detector.heatmap.checkpoint_sha256
    if a.bundle:result['bundle_sha256']=sha(a.bundle)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(dict(detections=len(predictions),seconds=result['seconds'],output=str(a.output))))


if __name__=='__main__':main()

"""Fast same-camera calibration from causal subpixel optical-flow correspondences."""
import time
import cv2
import numpy as np
from tracking.motion import CalibrationError, MotionModel, ViewGeometry


def calibrate_images(first, second, first_view, second_view, *, first_tick=0., second_tick=1.):
    if first_view.source_size != second_view.source_size:
        raise ValueError('Source dimensions changed')
    if second_tick <= first_tick:
        raise CalibrationError('Need a later moving image')
    from aligned_views import patches
    for image, view in ((first, first_view), (second, second_view)):
        if image is None or image.dtype != np.uint8 or image.shape[:2] != view.image_size[::-1]:
            raise ValueError('Image dimensions disagree with delivered view')
    gray, region, scale, size = patches(first, second, first_view, second_view, 96)
    if gray is None:
        raise CalibrationError('Insufficient overlapping image area')
    p = cv2.goodFeaturesToTrack(gray[0], maxCorners=1000, qualityLevel=.012, minDistance=8, blockSize=7)
    if p is None or len(p) < 30:
        raise CalibrationError('Insufficient background texture')
    params = dict(winSize=(21,21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,25,.01))
    q, st, _ = cv2.calcOpticalFlowPyrLK(gray[0],gray[1],p,None,**params)
    if q is None or st is None:
        raise CalibrationError('Forward optical flow failed')
    r, back, _ = cv2.calcOpticalFlowPyrLK(gray[1],gray[0],q,None,**params)
    if r is None or back is None:
        raise CalibrationError('Reverse optical flow failed')
    keep = st.ravel().astype(bool) & back.ravel().astype(bool) & (np.linalg.norm(r-p,axis=2).ravel()<.5)
    keep &= np.isfinite(q.reshape(-1,2)).all(1)
    pp, qq = p.reshape(-1,2)[keep], q.reshape(-1,2)[keep]
    if len(pp) < 30:
        raise CalibrationError('Too few consistent optical-flow features')
    x = (pp+.5)*scale-.5+region[:2]; y = (qq+.5)*scale-.5+region[:2]
    h, mask = cv2.findHomography(x,y,cv2.RANSAC,4.)
    if h is None or mask.sum()<25:
        raise CalibrationError('No coherent camera motion')
    # Exclude gross outliers, retaining mild height/parallax departures.
    hp = np.c_[x,np.ones(len(x))] @ h.T
    good = np.linalg.norm(hp[:,:2]/hp[:,2,None]-y,axis=1)<8.
    model = MotionModel.from_matches(x[good],y[good],source_size=first_view.source_size,first_tick=first_tick,second_tick=second_tick)
    info = dict(model.diagnostics, method='forward-backward-LK', matches=int(good.sum()), first_region=first_view.region, second_region=second_view.region)
    return MotionModel(model.matrix, model.source_size, model.origin_tick, model.calibrated_until, info)


def install():
    import tracking.workflow
    tracking.workflow.calibrate_images = calibrate_images

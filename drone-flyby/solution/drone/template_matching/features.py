"""Deterministic local-pixel-feature matching with geometric verification.

SIFT descriptors are computed from the supplied template pixels, not pretrained
weights. Multiple local matches must agree on a similarity transform. Image
upscaling only aids feature sampling; it does not recover unavailable detail.
"""
from dataclasses import dataclass, replace
import copy
import math

import cv2
import numpy as np

from .detector import TemplateDetector, Settings, nms, iou, features, correlation


def fuse_localizations(rows,threshold=.35,limit=300):
    """Fuse same-class pose alternatives, including nested foreground extents.

    Confidence is never increased by counting correlated template votes. Nearby
    distinct objects are kept unless their boxes overlap substantially.
    """
    groups=[]
    for row in sorted(rows,key=lambda r:(-r['score'],r['class'],r['bbox'],r['template_id'])):
        a=np.array(row['bbox'],float)
        group=None
        for candidate in groups:
            if candidate[0]['class']!=row['class']:
                continue
            b=np.array(candidate[0]['bbox'],float)
            intersection=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
            smaller=min(np.prod(a[2:]-a[:2]),np.prod(b[2:]-b[:2]))
            centres=np.linalg.norm((a[:2]+a[2:]-b[:2]-b[2:])/2)
            near=centres<.4*min(np.linalg.norm(a[2:]-a[:2]),np.linalg.norm(b[2:]-b[:2]))
            if iou(a,b)>threshold or (smaller>0 and intersection/smaller>.8 and near):
                group=candidate
                break
        if group is None:
            groups.append([row])
        else:
            group.append(row)
    fused=[]
    for group in groups:
        weights=np.array([r['score']/(1+r.get('reprojection_error',0)) for r in group])
        box=np.average([r['bbox'] for r in group],axis=0,weights=weights)
        fused.append({**group[0],'bbox':box.tolist(),'fused_hypotheses':len(group),
                      'fusion_template_ids':sorted({r['template_id'] for r in group})})
    return sorted(fused,key=lambda r:(-r['score'],r['class'],r['bbox'],r['template_id']))[:limit]


@dataclass(frozen=True)
class FeatureSettings:
    score_threshold: float = .8
    descriptor: str = 'sift'
    geometry: str = 'similarity'
    extent: str = 'source_aware'
    foreground_padding: float = .1
    matching: str = 'per_template'
    template_scales: tuple = (1.,)
    upsample: float = 2.
    ratio: float = .8
    distance_limit: float = 330.
    min_matches: int = 3
    min_inliers: int = 3
    reprojection_pixels: float = 2.5
    ransac_iterations: int = 3000
    min_scale: float = .35
    max_scale: float = 2.8
    max_instances: int = 8
    nms_iou: float = .35
    max_detections: int = 300

    def __post_init__(self):
        if not 0<=self.score_threshold<=1:
            raise ValueError('Invalid score threshold')
        if self.descriptor not in ('sift','root') or self.geometry not in ('similarity','affine') or self.extent not in ('rectangle','foreground','source_aware'):
            raise ValueError('Invalid feature model')
        if self.matching not in ('per_template','global','global_multiclass'):
            raise ValueError('Invalid feature matching mode')
        if not self.template_scales or any(not math.isfinite(s) or not 0<s<=1 for s in self.template_scales):
            raise ValueError('Template scales must be in (0,1]')
        if not 0<=self.foreground_padding<=.5:
            raise ValueError('Invalid foreground extent padding')
        if self.upsample <= 0 or not math.isfinite(self.upsample):
            raise ValueError('Invalid upsampling factor')
        if not 0 < self.ratio < 1 or self.distance_limit <= 0:
            raise ValueError('Invalid descriptor thresholds')
        if min(self.min_matches,self.min_inliers) < 3 or self.max_instances < 1:
            raise ValueError('At least three point matches are required')
        if not isinstance(self.ransac_iterations,int) or self.ransac_iterations<1:
            raise ValueError('Invalid RANSAC iteration limit')
        if not 0 < self.min_scale <= self.max_scale or self.reprojection_pixels <= 0:
            raise ValueError('Invalid geometric thresholds')


class FeatureDetector(TemplateDetector):
    def descriptor_transform(self,descriptors):
        if self.settings.descriptor=='root':
            return np.sqrt(descriptors/np.maximum(descriptors.sum(axis=1,keepdims=True),1e-12)).astype(np.float32)
        return descriptors

    def __init__(self, bank, settings=None):
        super().__init__(bank, Settings())
        self.settings = settings or FeatureSettings()
        self.sift = cv2.SIFT_create(nfeatures=18000,contrastThreshold=.012,edgeThreshold=15,sigma=1.2)
        self.feature_bank=[]
        self.appearance_templates={}
        for row,image,mask in self.templates:
            if row.get('mask_fallback'):
                continue
            gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            if mask is None:
                h,w=image.shape[:2]
                hull=np.float32([[0,0],[w,0],[w,h],[0,h]])
            else:
                # An annotation rectangle can include a second object or a large
                # background margin. Match and project the primary foreground.
                count,components,stats,_=cv2.connectedComponentsWithStats((mask>0).astype(np.uint8))
                component=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))
                mask=(components==component).astype(np.uint8)*255
                contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                hull=cv2.convexHull(max(contours,key=cv2.contourArea)).reshape(-1,2).astype(np.float32)+.5
            if self.settings.extent=='rectangle' or (self.settings.extent=='source_aware' and (row.get('source')=='organizer_reference' or row.get('annotation_extent'))):
                h,w=image.shape[:2]
                x1,y1,x2,y2=row.get('annotation_box',[0,0,w,h])
                hull=np.float32([[x1,y1],[x2,y1],[x2,y2],[x1,y2]])
            for reduction in self.settings.template_scales:
                rw,rh=[max(4,round(s*reduction)) for s in image.shape[1::-1]]
                # Downsample BEFORE feature upsampling: this models loss of pixel
                # detail at smaller projected object sizes instead of just resizing descriptors.
                reduced=cv2.resize(gray,(rw,rh),interpolation=cv2.INTER_AREA)
                enlarged=cv2.resize(reduced,None,fx=self.settings.upsample,fy=self.settings.upsample,interpolation=cv2.INTER_CUBIC)
                resized_mask=None if mask is None else cv2.resize(mask,(enlarged.shape[1],enlarged.shape[0]),interpolation=cv2.INTER_NEAREST)
                keypoints,descriptors=self.sift.detectAndCompute(enlarged,resized_mask)
                if descriptors is None or len(keypoints)<self.settings.min_matches:
                    continue
                factor=np.array([image.shape[1]/enlarged.shape[1],image.shape[0]/enlarged.shape[0]])
                # Pixel-centre mapping preserves offsets under noninteger resize.
                points=(np.float32([p.pt for p in keypoints])+.5)*factor
                variant={**row,'id':row['id']+f'@{reduction:g}','template_resolution':reduction}
                self.feature_bank.append((variant,image.shape[1::-1],points.astype(np.float32),self.descriptor_transform(descriptors),hull))
                self.appearance_templates[variant['id']]=(image,mask)
        if not self.feature_bank:
            raise ValueError('No templates contain sufficient local features')
        self._index=None

    def global_matches(self,descriptors,distance_limit):
        """Index all views together; compare distinct classes rather than duplicate views."""
        if self._index is None:
            metadata=[]
            for index,(row,_,_,desc,_) in enumerate(self.feature_bank):
                metadata.extend((index,point,row['class']) for point in range(len(desc)))
            self._metadata=metadata
            cv2.setRNGSeed(0)
            self._index=cv2.FlannBasedMatcher(dict(algorithm=1,trees=4),dict(checks=96))
            self._index.add([np.concatenate([r[3] for r in self.feature_bank])])
            self._index.train()
        groups=[[] for _ in self.feature_bank]
        for candidates in self._index.knnMatch(descriptors,k=min(16,len(self._metadata))):
            best=candidates[0]
            label=self._metadata[best.trainIdx][2]
            other=next((p.distance for p in candidates if self._metadata[p.trainIdx][2]!=label),None)
            multiclass=self.settings.matching=='global_multiclass'
            if best.distance>=distance_limit or (not multiclass and other is not None and best.distance>=self.settings.ratio*other):
                continue
            used=set()
            for match in candidates:
                index,point,cls=self._metadata[match.trainIdx]
                if (not multiclass and cls!=label) or index in used or match.distance>min(distance_limit,best.distance*1.15+1e-6):
                    continue
                groups[index].append(cv2.DMatch(match.queryIdx,point,match.distance))
                used.add(index)
                if len(used)>=6:
                    break
        return groups

    def scene_features(self,image):
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        gray=cv2.resize(gray,None,fx=self.settings.upsample,fy=self.settings.upsample,interpolation=cv2.INTER_CUBIC)
        return self.sift.detectAndCompute(gray,None)

    def detect(self,image,pixels_per_source_pixel=1., *, _features=None):
        if image is None or image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError('Expected uint8 BGR image')
        if not math.isfinite(pixels_per_source_pixel) or pixels_per_source_pixel <= 0:
            raise ValueError('Invalid image scale')
        if not self.feature_bank:
            return []
        cfg=self.settings
        keypoints,desc=self.scene_features(image) if _features is None else _features
        if desc is None or len(desc)<2:
            return []
        scene=(np.float32([k.pt for k in keypoints])+.5)/cfg.upsample
        desc=self.descriptor_transform(desc)
        distance_limit=.65 if cfg.descriptor=='root' else cfg.distance_limit
        matcher=cv2.BFMatcher(cv2.NORM_L2)
        groups=self.global_matches(desc,distance_limit) if cfg.matching.startswith('global') else None
        detections=[]
        for template_index,(row,(w,h),points,templates,hull) in enumerate(self.feature_bank):
            # Scene-to-template queries allow multiple scene instances to reuse
            # the same template features. Reverse matching would suppress them.
            if groups is None:
                pairs=matcher.knnMatch(desc,templates,k=2)
                good=[a for a,b in pairs if a.distance<cfg.ratio*b.distance and a.distance<distance_limit]
            else:
                good=groups[template_index]
            if len(good)<cfg.min_matches:
                continue
            remaining=list(range(len(good)))
            source=np.float32([points[m.trainIdx] for m in good])
            target=scene[[m.queryIdx for m in good]]
            for _ in range(cfg.max_instances):
                if len(remaining)<cfg.min_matches:
                    break
                cv2.setRNGSeed(0)
                estimate=cv2.estimateAffine2D if cfg.geometry=='affine' else cv2.estimateAffinePartial2D
                transform,inliers=estimate(source[remaining],target[remaining],
                    method=cv2.RANSAC,ransacReprojThreshold=cfg.reprojection_pixels,maxIters=cfg.ransac_iterations,confidence=.995,refineIters=10)
                if transform is None:
                    break
                chosen=[i for i,ok in zip(remaining,inliers.ravel()) if ok]
                # Peel this hypothesis even when it fails verification.
                remaining=[i for i in remaining if i not in set(chosen)]
                minimum=max(cfg.min_inliers,4) if cfg.geometry=='affine' else cfg.min_inliers
                if len(chosen)<minimum:
                    continue
                unique={good[i].trainIdx for i in chosen}
                if len(unique)<minimum:
                    continue
                scale=float(np.linalg.norm(transform[:,0]))
                singular=np.linalg.svd(transform[:,:2],compute_uv=False)
                if singular.min()<=0 or singular.max()/singular.min()>3 or np.linalg.det(transform[:,:2])<=0:
                    continue
                if not cfg.min_scale<=scale/pixels_per_source_pixel<=cfg.max_scale:
                    continue
                # Reject degenerate fits resting on several orientations of one corner.
                sp=source[chosen]
                covariance=np.cov(sp.T)
                if np.linalg.eigvalsh(covariance).min()<1.:
                    continue
                corners=cv2.transform(hull[None],transform)[0]
                low,high=corners.min(0),corners.max(0)
                foreground=cfg.extent=='foreground' or (cfg.extent=='source_aware' and row.get('source')!='organizer_reference' and not row.get('annotation_extent'))
                if foreground:
                    # Contours contain pixel centres. Cover the transformed pixel
                    # footprint before adding the segmentation uncertainty margin.
                    pixel_radius=.5*np.abs(transform[:,:2]).sum(axis=1)
                    low,high=low-pixel_radius,high+pixel_radius
                    margin=(high-low)*cfg.foreground_padding
                    low,high=low-margin,high+margin
                if np.any(low<0) or high[0]>image.shape[1] or high[1]>image.shape[0]:
                    continue
                projected=cv2.transform(sp[None],transform)[0]
                error=float(np.median(np.linalg.norm(projected-target[chosen],axis=1)))
                coverage=float(np.ptp(sp[:,0])*np.ptp(sp[:,1])/(w*h))
                if coverage<.015:
                    continue
                distance=float(np.median([good[i].distance for i in chosen]))
                score=float(np.clip(.5+.045*min(len(unique),8)+.12*(1-distance/distance_limit)-.02*error,0,1))
                if score<cfg.score_threshold:
                    continue
                detections.append({'class':row['class'],'bbox':[float(low[0]),float(low[1]),float(high[0]),float(high[1])],
                    'score':score,'template_id':row['id'],'scale':scale/pixels_per_source_pixel,
                    'angle':float(np.degrees(np.arctan2(transform[1,0],transform[0,0]))),
                    'inliers':len(unique),'reprojection_error':error,'feature_coverage':coverage,'descriptor_distance':distance,
                    'template_transform':transform.tolist()})
        return nms(detections,cfg.nms_iou,cfg.max_detections)

    def appearance_similarity(self,image,row):
        """Check pixels after the independently estimated feature transform."""
        template,mask=self.appearance_templates[row['template_id']]
        valid=np.ones(template.shape[:2],bool) if mask is None else mask>0
        aligned=cv2.warpAffine(image,np.asarray(row['template_transform']),template.shape[1::-1],flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP)
        gray,high=features(template);ag,ah=features(aligned)
        return float(.45*correlation(gray[valid],ag[valid])+.55*correlation(high[valid],ah[valid]))


@dataclass(frozen=True)
class EnsembleSettings:
    score_threshold: float = .8
    foreground_padding: float = .1
    nms_iou: float = .35
    max_detections: int = 300
    matching: str = 'global'
    template_scales: tuple = (1.,)
    upsample: float = 2.
    ransac_iterations: int = 3000
    calibrated_fallback: bool = False
    strong_threshold: float | None = None

    def __post_init__(self):
        FeatureSettings(score_threshold=self.score_threshold,foreground_padding=self.foreground_padding,matching=self.matching,template_scales=self.template_scales,upsample=self.upsample,ransac_iterations=self.ransac_iterations)
        if not 0<=self.nms_iou<=1 or self.max_detections<1:
            raise ValueError('Invalid suppression settings')
        if self.strong_threshold is not None and not self.score_threshold<=self.strong_threshold<=1:
            raise ValueError('Strong threshold must exceed proposal threshold')


class FeatureEnsemble(TemplateDetector):
    """Similarity and affine matches share image features, then class-wise NMS."""
    def __init__(self,bank,settings=None):
        self.settings=settings or EnsembleSettings()
        cfg=self.settings
        self.matchers=[FeatureDetector(bank,FeatureSettings(score_threshold=cfg.score_threshold,
                       foreground_padding=cfg.foreground_padding,matching=cfg.matching,template_scales=cfg.template_scales,upsample=cfg.upsample,ransac_iterations=cfg.ransac_iterations,descriptor=descriptor,geometry=geometry))
                       for descriptor,geometry in [('sift','similarity'),('root','affine')]]
        self.manifest=self.matchers[0].manifest
        self.templates=self.matchers[0].templates
        self.calibrated_matchers=[]
        if cfg.calibrated_fallback:
            # Repeated asset parts compete in the global descriptor index.
            # Verify each disclosed calibration pose independently as a fallback,
            # reusing the same scene keypoints and strict geometric checks.
            for matcher in self.matchers:
                fallback=copy.copy(matcher)
                fallback.settings=replace(matcher.settings,matching='per_template')
                fallback.feature_bank=[r for r in matcher.feature_bank if r[0].get('calibration')]
                self.calibrated_matchers.append(fallback)

    def detect(self,image,pixels_per_source_pixel=1.):
        if image is None or image.ndim!=3 or image.shape[2]!=3 or image.dtype!=np.uint8:
            raise ValueError('Expected uint8 BGR image')
        extracted=self.matchers[0].scene_features(image)
        rows=[]
        for matcher in self.matchers:
            for row in matcher.detect(image,pixels_per_source_pixel,_features=extracted):
                rows.append({**row,'matcher':matcher.settings.descriptor+'-'+matcher.settings.geometry})
        for matcher in self.calibrated_matchers:
            for row in matcher.detect(image,pixels_per_source_pixel,_features=extracted):
                rows.append({**row,'matcher':matcher.settings.descriptor+'-'+matcher.settings.geometry,
                             'calibrated_fallback':True})
        return self.suppress(rows)

    def _suppress_group(self,rows):
        primary=fuse_localizations([r for r in rows if not r.get('calibrated_fallback')],self.settings.nms_iou,self.settings.max_detections)
        fallback=fuse_localizations([r for r in rows if r.get('calibrated_fallback')],self.settings.nms_iou,self.settings.max_detections)
        return (primary+[r for r in fallback if not any(r['class']==p['class'] and iou(r['bbox'],p['bbox'])>self.settings.nms_iou for p in primary)])[:self.settings.max_detections]

    def suppress(self,rows):
        threshold=self.settings.strong_threshold
        if threshold is None:return self._suppress_group(rows)
        strong=self._suppress_group([r for r in rows if r['score']>=threshold])
        weak=self._suppress_group([r for r in rows if r['score']<threshold])
        return (strong+[r for r in weak if not any(r['class']==p['class'] and iou(r['bbox'],p['bbox'])>self.settings.nms_iou for p in strong)])[:self.settings.max_detections]

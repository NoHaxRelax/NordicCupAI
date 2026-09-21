import unittest,numpy as np
from tracking import Detection,DroneTrackingWorkflow,RevisitTracker,RevisitConfig,MotionModel,ViewGeometry
from tracking.test_revisit import request
from local_evaluator import Camera
class InspectionTests(unittest.TestCase):
 def setup_workflow(self,label='small_launcher',confidence=.4):
  a=np.zeros((3,3));a[1,2]=10
  w=DroneTrackingWorkflow(RevisitConfig(birth_confidence=.25,update_confidence=.15,miss_rule='seen'),revisit_every=8)
  w.tracker=RevisitTracker(MotionModel(a,(3840,2160),0.,1.),'test',w.config)
  req=request(8,Camera(1,960,540));view=ViewGeometry.from_request(req)
  w.tracker.update([Detection(label,(440,200,460,225),confidence)],view,8,8)
  return w,req
 def test_weak_small_object_is_emitted_and_gets_legal_l2_inspection(self):
  w,req=self.setup_workflow();self.assertEqual(len(w.tracker.response(req)['annotations']),1)
  focus=w.uncertain_small_track(req,8);self.assertIsNotNone(focus)
  c=Camera(1,960,540);c.apply(**w.camera.next_view(req,focus_box=focus));self.assertEqual(c.resolution_level,2)
  self.assertIsNone(w.uncertain_small_track(req,8))
 def test_confident_small_object_and_aircraft_do_not_force_l2(self):
  for label,confidence in [('small_launcher',.8),('medium_plane',.4),('helicopter',.4)]:
   w,req=self.setup_workflow(label,confidence);self.assertIsNone(w.uncertain_small_track(req,8))
   self.assertTrue(w.tracker.response(req)['annotations'])
 def test_low_resolution_miss_does_not_retire_a_native_small_track(self):
  w,req=self.setup_workflow();t=next(iter(w.tracker.tracks.values()));t.seen_pixels=40
  view=ViewGeometry((3840,2160),(0,0,3840,2160),(960,540))
  for f in range(9,15):w.tracker.update([],view,f,f)
  self.assertEqual(t.visible_misses,0);self.assertEqual(len(w.tracker.tracks),1)

"""Delayed transport must not roll back a live tracker or run stale inference."""
import os,unittest
from unittest.mock import Mock,patch
os.environ['DRONE_DETECTOR']='none'
import example
from dtos import DroneFlybyPredictRequestDto
from local_evaluator import Camera,build_request

class EndpointOrderTests(unittest.TestCase):
    def test_stale_frame_is_empty_and_duplicate_reuses_cached_answer(self):
        payload=build_request(3,2,Camera(),'not-decoded-for-stale-request',None)
        req=DroneFlybyPredictRequestDto.model_validate(payload)
        session=example.Session(req.sequence_id)
        session.workflow.last_frame=10
        session.workflow.last_request_id='newer-request'
        session.workflow.last_response=dict(request_id='newer-request',frame=11,annotations=[],requested_view=None)
        with patch.object(example,'_session',return_value=session),patch.object(example,'DETECTOR',Mock(side_effect=AssertionError('Stale pixels reached inference'))):
            answer=example.predict(req)
            self.assertEqual(answer.frame,3);self.assertFalse(answer.annotations)
            self.assertEqual(session.workflow.last_frame,10)
            session.workflow.last_frame=2;session.workflow.last_request_id=req.request_id
            session.workflow.last_response=dict(request_id=req.request_id,frame=3,annotations=[],requested_view=None)
            answer=example.predict(req)
            self.assertEqual(answer.model_dump(),session.workflow.last_response)

if __name__=='__main__':unittest.main()

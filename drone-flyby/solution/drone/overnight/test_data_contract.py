"""CPU tests for supervision boundaries, split leakage, and release gates."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from common import covered, frame_split, read, release_errors, sha, verify_dataset, write
from prepare import load_sources, windows
from manage import queue


class DataContractTests(unittest.TestCase):
    def test_union_requires_every_pixel(self):
        self.assertTrue(covered([0,0,1920,540],[[0,0,960,540],[960,0,1920,540]]))
        self.assertFalse(covered([0,0,1920,541],[[0,0,960,540],[960,0,1920,540]]))
        self.assertFalse(covered([0,0,1920,540],[[0,0,959,540],[960,0,1920,540]]))

    def test_zoom_strips_stay_inside_reviewed_band(self):
        for z in range(3):
            boxes=list(windows([0,0,3840,540],z))
            self.assertEqual(boxes[0][0],0)
            self.assertEqual(boxes[-1][2],3840)
            self.assertTrue(all(covered(b,[[0,0,3840,540]]) for b in boxes))
            self.assertTrue(all(b[3]==540 for b in boxes))

    def test_buffer_and_release_fail_closed(self):
        c=read(Path(__file__).with_name('config.json'))
        self.assertEqual(frame_split(149,c),'train')
        self.assertIsNone(frame_split(150,c))
        self.assertIsNone(frame_split(180,c))
        self.assertEqual(frame_split(181,c),'dev')
        self.assertEqual(len(release_errors({'source_fingerprint':'a'},{})),4)
        release=dict(annotations_frozen=True,validation_training_permitted=True,launch_enabled=True,source_fingerprint='a')
        self.assertFalse(release_errors({'source_fingerprint':'a'},release))
        self.assertTrue(release_errors({'source_fingerprint':'b'},release))

    def test_empty_reviewed_frames_and_cross_split_tracks(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);config=read(Path(__file__).with_name('config.json'))
            write(root/'data/drone/reference/helsinki/annotations/frame_000000.json',
                  dict(frame=0,annotations=[dict(object_id='tank',bbox=[1,1,5,5])]))
            write(root/config['validation_annotation_dir']/'one-track.json',dict(annotations=[
                dict(frame=f,**{'class':'tank'},bbox_source_xyxy=[1,1,5,5],review_status='directly_reviewed') for f in [149,181]]))
            sheet=dict(frame=185,source_regions_xyxy=[[0,0,960,540]],primary_review={'status':'reviewed_empty'},small_object_review={'status':'reviewed_empty'})
            write(root/'artifacts/drone-validation-coverage/coverage-ledger.json',
                dict(completion={'safe_for_reviewed_region_negative_training':True},sheets=[sheet]))
            frames,regions,crossing,_=load_sources(root,config)
            self.assertEqual(crossing,{'one-track'})
            self.assertEqual(frames['validation',185]['annotations'],[])
            self.assertEqual(frames['validation',185]['split'],'dev')
            self.assertEqual(regions[185],[[0,0,960,540]])
            path=root/config['validation_annotation_dir']/'one-track.json'
            doc=read(path);doc['annotations'][1]['review_status']='algorithmic_projected';write(path,doc)
            config['split_reviewed_occurrences_only']=True
            frames,_,crossing,_=load_sources(root,config)
            self.assertEqual(crossing,set())
            self.assertTrue(frames['validation',149]['annotations'][0]['reviewed'])
            self.assertFalse(frames['validation',181]['annotations'][0]['reviewed'])

    def test_corrected_export_supersedes_rejected_raw_track(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);config=read(Path(__file__).with_name('config.json'))
            annotation=dict(frame=109,**{'class':'medium_plane'},bbox_source_xyxy=[1,1,5,5],review_status='directly_reviewed_track')
            write(root/'data/drone/reference/helsinki/annotations/frame_000000.json',dict(frame=0,annotations=[]))
            write(root/'data/drone/training/manual-validation/rejected.json',dict(annotations=[annotation]))
            write(root/config['validation_annotation_dir']/'rejected.json',dict(annotations=[],rejection={'status':'rejected_false_track_native_resolution_audit'}))
            write(root/config['validation_annotation_dir']/'projected.json',dict(annotations=[dict(annotation,review_status='algorithmic_projected')]))
            write(root/'artifacts/drone-validation-coverage/coverage-ledger.json',dict(completion={'safe_for_reviewed_region_negative_training':True},sheets=[]))
            frames,_,_,hashes=load_sources(root,config)
            self.assertEqual(len(frames['validation',109]['annotations']),1)
            self.assertFalse(frames['validation',109]['annotations'][0]['reviewed'])
            self.assertTrue(all('manual-validation' not in path for path in hashes))

    def test_unreviewed_sheet_never_creates_background(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);config=read(Path(__file__).with_name('config.json'))
            write(root/'data/drone/reference/helsinki/annotations/frame_000000.json',dict(frame=0,annotations=[]))
            write(root/'artifacts/drone-validation-coverage/coverage-ledger.json',dict(completion={'safe_for_reviewed_region_negative_training':True},
                sheets=[dict(frame=181,source_regions_xyxy=[[0,0,960,540]],primary_review={'status':'reviewed_empty'},small_object_review={'status':'pending'})]))
            frames,regions,_,_=load_sources(root,config)
            self.assertNotIn(181,regions)

    def test_content_hash_verification_catches_changed_labels(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);(root/'a.png').write_bytes(b'image');(root/'a.txt').write_text('0 0.5 0.5 0.1 0.1')
            m={'records':[dict(file='a.png',sha256=sha(root/'a.png'),label_file='a.txt',label_sha256=sha(root/'a.txt'))]}
            verify_dataset(root,m)
            (root/'a.txt').write_text('')
            with self.assertRaises(ValueError):verify_dataset(root,m)

    def test_queue_gates_and_deadline_do_not_start_a_process(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);data=root/'data'
            write(data/'manifest.json',dict(source_fingerprint='test',records=[],config={'experiments':[{'id':'one'}]}))
            release=root/'release.json';write(release,{})
            args=SimpleNamespace(data=data,release=release,run=root/'runs'/'one',host='mypc',experiments=None,weights=root/'weights')
            with patch('manage.platform.node',return_value='test-host'), patch('manage.subprocess.Popen') as process:
                with self.assertRaises(SystemExit):queue(args)
                self.assertFalse(args.run.exists());process.assert_not_called()
                write(release,dict(source_fingerprint='test',annotations_frozen=True,validation_training_permitted=True,
                    launch_enabled=True,stop_starting_jobs_at='2000-01-01T00:00:00+00:00'))
                self.assertEqual(queue(args),0);process.assert_not_called()
                self.assertEqual(read(args.run/'status.json')['state'],'window_finished')
                self.assertFalse((args.run.parent/'overnight-mypc.gpu.lock').exists())

    def test_second_run_name_cannot_bypass_shared_gpu_lock(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);data=root/'data';release=root/'release.json'
            write(data/'manifest.json',dict(source_fingerprint='test',records=[],config={'experiments':[{'id':'one'}]}))
            write(release,dict(source_fingerprint='test',annotations_frozen=True,validation_training_permitted=True,launch_enabled=True))
            lock=root/'runs'/'overnight-mypc.gpu.lock';lock.mkdir(parents=True)
            write(lock/'owner.json',dict(pid=123,host='mypc',run='another-run'))
            args=SimpleNamespace(data=data,release=release,run=root/'runs'/'new-name',host='mypc',experiments=None,weights=root/'weights')
            with patch('manage.platform.node',return_value='test-host'), patch('manage.subprocess.Popen') as process:
                with self.assertRaises(SystemExit):queue(args)
                process.assert_not_called();self.assertTrue(lock.exists())


if __name__=='__main__':unittest.main()

import unittest
from .yolo_thresholds import score


class ThresholdTests(unittest.TestCase):
    def test_class_confusion_and_duplicate_objects(self):
        annotation = dict(fully_contained=True, class_id=2, class_name='target',
                          group='track', bbox_xyxy=[0, 0, 10, 10])
        rows = [dict(kind='positive', zoom=0, annotations=[annotation, annotation]),
                dict(kind='background', zoom=0, annotations=[])]
        detections = [[dict(class_id=1, score=.1, box=[0, 0, 10, 10])],
                      [dict(class_id=1, score=.02, box=[0, 0, 10, 10])]]
        self.assertEqual(score(rows, detections, .03, True)['recall_per_class']['target'], 0)
        result = score(rows, detections, .03, False)
        self.assertEqual(result['recall_per_class']['target'], .5)
        self.assertEqual(result['background_false_positive_rate'], 0)
        self.assertEqual(score(rows, detections, .001, False)['background_false_positive_rate'], 1)


if __name__ == '__main__':
    unittest.main()

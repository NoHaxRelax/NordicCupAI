import unittest

from evaluate_views import metrics


class ViewMetricsTests(unittest.TestCase):
    def test_one_target_cannot_count_twice_and_wrong_class_is_false_positive(self):
        box=[0,0,10,10]
        rows=[dict(truth=[{'class':'tank','bbox':box}],predictions=[
            {'class':'tank','bbox':box,'score':.9},
            {'class':'tank','bbox':box,'score':.8},
            {'class':'plane','bbox':box,'score':.7}])]
        result=metrics(rows,['tank','plane'],.5)
        self.assertEqual((result['tp'],result['fp'],result['targets']),(1,2,1))
        self.assertEqual(result['missing_classes'],['plane'])

    def test_empty_complete_region_counts_false_positives(self):
        rows=[dict(truth=[],predictions=[{'class':'tank','bbox':[0,0,1,1],'score':.6}])]
        result=metrics(rows,['tank'],.5)
        self.assertEqual(result['fp'],1)
        self.assertIsNone(result['recall'])
        self.assertEqual(metrics(rows,['tank'],.7)['fp'],0)

    def test_class_agnostic_separates_localization_from_classification(self):
        box=[0,0,10,10]
        rows=[dict(truth=[{'class':'tank','bbox':box}],predictions=[{'class':'plane','bbox':box,'score':.9}])]
        self.assertEqual(metrics(rows,['tank','plane'],.5)['tp'],0)
        self.assertEqual(metrics(rows,['tank','plane'],.5,True)['tp'],1)


if __name__=='__main__':unittest.main()

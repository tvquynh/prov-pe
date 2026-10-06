"""Small scientific-software tests; no classifier fits and no real-score change."""
import unittest
import numpy as np
from reproduce_revision import fixture_check,require
from analyze_revision import metric
from revision_common import threshold

class VerificationTests(unittest.TestCase):
    def test_ties_respect_budget(self):
        s=np.array([.1,.2,.2,.2,.4]);t=threshold(s,.4)
        self.assertEqual(t,.4);self.assertEqual(int((s>=t).sum()),1)
    def test_zero_budget_and_positive_above_negative_maximum(self):
        t=threshold(np.array([.2,.8]),0)
        self.assertGreater(t,.8);self.assertTrue(.9>=t)
    def test_weighted_fpr_and_undefined_positive_metrics(self):
        r=metric(np.array([0,0,0]),np.array([.9,.1,.1]),.5,np.array([2.,1.,1.]))
        self.assertEqual(r['FPR'],.5);self.assertIsNone(r['TPR']);self.assertIsNone(r['ROC_AUC']);self.assertIsNone(r['AP'])
    def test_label_cutoff_fixed_negatives_implies_fixed_fpr(self):
        y=np.array([0,0,1,1]);s=np.array([.1,.8,.3,.9])
        self.assertEqual(metric(y,s,.5)['FPR'],metric(y[[0,1,3]],s[[0,1,3]],.5)['FPR'])
    def test_accept_roundoff_without_decision_change(self):
        r=fixture_check(np.array([.1+1e-14,.8]),np.array([.1,.8]),[.5],1e-12)
        self.assertEqual(r['decision_disagreements'],{'.5'.replace('.5','0.5'):0})
    def test_reject_decision_change_within_tolerance(self):
        with self.assertRaises(RuntimeError):fixture_check(np.array([.5-1e-14]),np.array([.5]),[.5],1e-12)
    def test_reject_excess_error(self):
        with self.assertRaises(RuntimeError):fixture_check(np.array([.1]),np.array([.2]),[.5],1e-12)
    def test_reject_nonfinite(self):
        with self.assertRaises(RuntimeError):fixture_check(np.array([np.nan]),np.array([.2]),[.5],1e-12)
    def test_reject_shape(self):
        with self.assertRaises(RuntimeError):fixture_check(np.array([.1,.2]),np.array([.2]),[.5],1e-12)
    def test_require_survives_python_optimization(self):
        with self.assertRaises(RuntimeError):require(False,'deliberate invalid input')

if __name__=='__main__':unittest.main(verbosity=2)

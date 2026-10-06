"""Synthetic regression tests for checker semantics; no study models or data."""
from pathlib import Path
import argparse
import json
import unittest
import numpy as np
from fixture_profiles import compare_fixture


class FixtureProfiles(unittest.TestCase):
    def check(self, a, b, profile='numerical'):
        return compare_fixture(a, b, {'0.01': .5, '0.001': .9}, profile)

    def test_exact_rejects_one_bit(self):
        b = np.array([.2, .8]); a = b.copy(); a[0] = np.nextafter(a[0], 1)
        self.assertEqual(self.check(a, b, 'exact')['status'], 'FAIL')

    def test_numerical_accepts_tiny_unchanged_decisions(self):
        b = np.array([.2, .8]); a = b + 1e-14
        result = self.check(a, b)
        self.assertEqual(result['status'], 'PASS'); self.assertFalse(result['exact_bitwise'])

    def test_excess_error_fails(self):
        self.assertEqual(self.check(np.array([.2 + 2e-12]), np.array([.2]))['status'], 'FAIL')

    def test_decision_flip_inside_tolerance_fails(self):
        result = self.check(np.array([.5]), np.array([np.nextafter(.5, 0)]))
        self.assertEqual(result['status'], 'FAIL'); self.assertEqual(result['decision_disagreements']['0.01'], 1)

    def test_shape_fails(self):
        self.assertEqual(self.check(np.array([[.2]]), np.array([.2]))['status'], 'FAIL')

    def test_nan_fails(self):
        self.assertEqual(self.check(np.array([np.nan]), np.array([.2]))['status'], 'FAIL')

    def test_exact_pass(self):
        self.assertEqual(self.check(np.array([.2, .9]), np.array([.2, .9]), 'exact')['status'], 'PASS')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Fresh output file required')
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(FixtureProfiles))
    args.output.write_text(json.dumps({'status': 'PASS' if result.wasSuccessful() else 'FAIL', 'tests': result.testsRun,
                                      'failures': len(result.failures), 'errors': len(result.errors), 'synthetic_only': True,
                                      'optimization_enabled': not __debug__}, indent=2), encoding='utf-8')
    raise SystemExit(0 if result.wasSuccessful() else 1)

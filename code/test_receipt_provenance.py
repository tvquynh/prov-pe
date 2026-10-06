"""Metadata-only tests; no study data, model inference or metric recomputation."""
from pathlib import Path
import argparse
import json
import sys
import unittest
from unittest.mock import patch

import reproduce_review


class ReceiptProvenance(unittest.TestCase):
    def test_base_runtime_reports_observed_prefixes(self):
        with patch.object(sys, 'prefix', '/observed/base'), patch.object(sys, 'base_prefix', '/observed/base'):
            result = reproduce_review.observed_environment([], Path('metadata_output'))
        self.assertEqual(result['python_prefix'], '/observed/base')
        self.assertEqual(result['python_base_prefix'], '/observed/base')
        self.assertFalse(result['prefix_differs_from_base'])

    def test_different_prefix_is_not_an_isolation_assurance(self):
        with patch.object(sys, 'prefix', '/observed/venv'), patch.object(sys, 'base_prefix', '/observed/base'):
            result = reproduce_review.observed_environment([], Path('metadata_output'))
        self.assertTrue(result['prefix_differs_from_base'])
        self.assertNotIn('isolation', result)
        for unmeasured_claim in ['dedicated_environment', 'inherited_packages_read_only', 'global_environment_unchanged']:
            self.assertNotIn(unmeasured_claim, result)

    def test_paths_and_runtime_are_direct_observations(self):
        output = Path('metadata_output')
        result = reproduce_review.observed_environment([], output)
        self.assertEqual(result['executable'], sys.executable)
        self.assertEqual(result['python'], sys.version)
        self.assertEqual(result['output_directory'], str(output.resolve()))
        self.assertEqual(result['python_prefix'], sys.prefix)
        self.assertEqual(result['python_base_prefix'], sys.base_prefix)
        self.assertEqual(result['prefix_differs_from_base'], sys.prefix != sys.base_prefix)
        self.assertIsInstance(result['platform'], str)
        json.dumps(result, allow_nan=False)

    def test_metadata_needs_no_scientific_execution(self):
        with patch.object(reproduce_review, 'load_dependencies', side_effect=RuntimeError('Dependency loading forbidden')), \
                patch.object(reproduce_review, 'main', side_effect=RuntimeError('Scientific execution forbidden')):
            result = reproduce_review.observed_environment(['pip'], Path('metadata_output'))
        self.assertIn('pip', result['versions'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Fresh output file required')
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ReceiptProvenance))
    report = {
        'status': 'PASS' if result.wasSuccessful() else 'FAIL',
        'tests': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
        'metadata_only': True, 'optimization_enabled': not __debug__,
        'actual_runtime_observations': reproduce_review.observed_environment(
            ['numpy', 'pandas', 'pyarrow', 'lightgbm', 'scikit-learn', 'scipy'], args.output.parent),
        'model_inference_calls': 0, 'scientific_metric_recomputations': 0,
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    raise SystemExit(0 if result.wasSuccessful() else 1)

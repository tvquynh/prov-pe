"""Exact or tightly bounded numerical validation of saved probability fixtures."""
ATOL = 1e-12


def compare_fixture(actual, expected, thresholds, profile):
    import numpy as np
    if profile not in ['exact', 'numerical']:
        raise ValueError('Profile must be exact or numerical')
    if len(thresholds) != 2 or not all(np.isfinite(x) for x in thresholds.values()):
        raise ValueError('Two finite frozen thresholds are required')
    actual = np.asarray(actual); expected = np.asarray(expected)
    result = {'profile': profile, 'actual_shape': list(actual.shape), 'expected_shape': list(expected.shape),
              'shape_match': actual.shape == expected.shape and actual.ndim == 1,
              'actual_dtype': str(actual.dtype), 'expected_dtype': str(expected.dtype),
              'finite': bool(np.isfinite(actual).all() and np.isfinite(expected).all()),
              'atol': ATOL if profile == 'numerical' else 0.0, 'rtol': 0.0,
              'exact_bitwise': False, 'max_absolute_error': None,
              'decision_disagreements': {}, 'status': 'FAIL'}
    if not result['shape_match'] or not result['finite'] or actual.size == 0:
        result['reason'] = 'INVALID_SHAPE_OR_NONFINITE'
        return result
    result['probability_range_valid'] = bool(((actual >= 0) & (actual <= 1)).all() and ((expected >= 0) & (expected <= 1)).all())
    result['exact_bitwise'] = actual.dtype == expected.dtype and actual.tobytes() == expected.tobytes()
    result['max_absolute_error'] = float(np.max(np.abs(actual - expected)))
    result['decision_disagreements'] = {str(k): int(np.count_nonzero((actual >= v) != (expected >= v))) for k, v in thresholds.items()}
    score_match = result['exact_bitwise'] if profile == 'exact' else result['max_absolute_error'] <= ATOL
    if score_match and not any(result['decision_disagreements'].values()) and result['probability_range_valid']:
        result['status'] = 'PASS'
    else:
        result['reason'] = 'SCORE_OR_DECISION_MISMATCH'
    return result

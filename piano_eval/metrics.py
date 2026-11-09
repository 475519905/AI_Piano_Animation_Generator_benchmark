"""Independent, model-free piano animation measurements. SI inputs; no alignment."""
import numpy as np


def numeric(value, name, ndim=None):
    a = np.asarray(value)
    if a.dtype.kind not in 'fiu' or (ndim is not None and a.ndim != ndim):
        raise ValueError(f'{name}: expected a numeric array with ndim={ndim}')
    a = a.astype(np.float64, copy=True)
    if not np.isfinite(a).all():
        raise ValueError(f'{name}: non-finite values')
    return a


def summary(values, unit):
    a = numeric(values, 'values').reshape(-1)
    if not a.size:
        return {'status': 'not_measured', 'count': 0, 'unit': unit, 'reason': 'no eligible samples'}
    return {'status': 'measured', 'count': int(a.size), 'unit': unit,
            'mean': float(a.mean()), 'p50': float(np.percentile(a, 50)),
            'p95': float(np.percentile(a, 95)), 'max': float(a.max())}


def point_errors(prediction, reference):
    p = numeric(prediction, 'prediction', 3)
    r = numeric(reference, 'reference', 3)
    if p.shape != r.shape or p.shape[-1] != 3 or not p.shape[0] or not p.shape[1]:
        raise ValueError('point arrays must share nonempty (T,J,3) shape')
    delta = p - r
    distances = np.linalg.norm(delta, axis=-1)
    sse = float(np.sum(delta * delta))
    return {'status': 'measured', 'alignment': 'none',
            'coordinate_rmse_mm': float(np.sqrt(sse / delta.size) * 1000),
            'mean_point_error_mm': float(distances.mean() * 1000),
            'point_distance_mm': summary(distances * 1000, 'mm'),
            'squared_coordinate_error_sum_m2': sse, 'coordinate_count': int(delta.size),
            'point_distance_sum_m': float(distances.sum()), 'point_count': int(distances.size)}


def rotation_errors(prediction, reference):
    p = numeric(prediction, 'prediction rotations', 3)
    r = numeric(reference, 'reference rotations', 3)
    if p.shape != r.shape or p.shape[-1] != 4 or not p.shape[0] or not p.shape[1]:
        raise ValueError('quaternions must share nonempty (T,J,4) xyzw shape')
    pn, rn = np.linalg.norm(p, axis=-1, keepdims=True), np.linalg.norm(r, axis=-1, keepdims=True)
    if not np.isfinite(pn).all() or not np.isfinite(rn).all() or np.any(pn < 1e-12) or np.any(rn < 1e-12):
        raise ValueError('zero/near-zero quaternion is not a valid rotation')
    dot = np.abs(np.sum((p / pn) * (r / rn), axis=-1))
    angles = np.degrees(2 * np.arccos(np.clip(dot, 0, 1)))
    return summary(angles, 'degrees')


def integers(value, name):
    a = np.asarray(value)
    if a.dtype.kind not in 'iu':
        raise ValueError(f'{name}: integer labels required')
    return a.astype(np.int64, copy=False)


def fingering(prediction, reference):
    p, r = integers(prediction, 'predicted fingering'), integers(reference, 'reference fingering')
    if p.ndim != 1 or p.shape != r.shape:
        raise ValueError('fingering arrays must share (N,) shape')
    if np.any((r < 0) | (r > 9)) or np.any((p < -1) | (p > 9)):
        raise ValueError('reference must be 0..9; prediction may additionally use -1 abstention')
    if not r.size:
        return {'status': 'not_measured', 'reason': 'no reference events', 'events': 0}
    valid = p >= 0
    exact = int(np.count_nonzero(valid & (p == r)))
    hand = int(np.count_nonzero(valid & ((p // 5) == (r // 5))))
    confusion = np.zeros((10, 11), dtype=np.int64)
    np.add.at(confusion, (r, np.where(valid, p, 10)), 1)
    return {'status': 'measured', 'events': int(r.size), 'predicted_events': int(valid.sum()),
            'exact_correct': exact, 'hand_correct': hand, 'exact_accuracy': exact / r.size,
            'hand_accuracy': hand / r.size, 'prediction_coverage': float(valid.mean()),
            'confusion_reference_rows_prediction_columns_0_to_9_then_abstain': confusion.tolist()}


def timing_errors(predicted_start_s, reference_start_s):
    p, r = numeric(predicted_start_s, 'predicted starts', 1), numeric(reference_start_s, 'reference starts', 1)
    if p.shape != r.shape:
        raise ValueError('event times must share shape; match event IDs first')
    delta = (p - r) * 1000
    result = summary(np.abs(delta), 'ms')
    if delta.size:
        result.update(signed_mean_ms=float(delta.mean()), absolute_error_sum_ms=float(np.abs(delta).sum()))
    return result


def temporal_motion(positions_m, fps, held_mask=None):
    p = numeric(positions_m, 'positions', 3)
    if not p.shape[0] or not p.shape[1] or p.shape[-1] != 3:
        raise ValueError('positions must be nonempty (T,J,3)')
    if isinstance(fps, bool) or not np.isfinite(fps) or fps <= 0:
        raise ValueError('fps must be positive and finite')
    out = {'status': 'measured', 'fps': float(fps), 'interpretation': 'descriptive; lower is not always better'}
    for order, name, unit in [(1, 'speed', 'mm/s'), (2, 'acceleration', 'mm/s^2'), (3, 'jerk', 'mm/s^3')]:
        delta = np.diff(p, n=order, axis=0) * (1000 * float(fps) ** order)
        out[name] = summary(np.linalg.norm(delta, axis=-1), unit)
    if held_mask is not None:
        h = np.asarray(held_mask)
        if h.dtype.kind != 'b' or h.shape != p.shape[:2]:
            raise ValueError('held mask must be boolean (T,J)')
        if len(h) < 4:
            out['held_jerk'] = summary(np.empty(0), 'mm/s^3')
            return out
        valid = h[3:].copy()
        for lag in (1, 2, 3):
            valid &= h[3-lag:len(h)-lag]
        jerk = np.linalg.norm(np.diff(p, n=3, axis=0), axis=-1) * (1000 * float(fps) ** 3)
        out['held_jerk'] = summary(jerk[valid], 'mm/s^3')
    else:
        out['held_jerk'] = {'status': 'not_measured', 'reason': 'no independent held-contact reference mask'}
    return out


def assigned_contact(tips_m, expected_mask, expected_key, key_bounds_m, tolerance_mm):
    """Reference-assigned point-to-key-box proxy, NOT full mesh collision."""
    tips = numeric(tips_m, 'fingertips', 3)
    mask = np.asarray(expected_mask)
    key = integers(expected_key, 'expected keys')
    bounds = numeric(key_bounds_m, 'key bounds')
    if tips.shape[1:] != (10, 3) or not tips.shape[0]:
        raise ValueError('fingertips must be nonempty (T,10,3)')
    if mask.dtype.kind != 'b' or mask.shape != tips.shape[:2] or key.shape != mask.shape:
        raise ValueError('reference contact mask/keys must be (T,10)')
    if bounds.ndim not in (3, 4) or bounds.shape[-2:] != (2, 3):
        raise ValueError('key bounds must be (K,2,3) or (T,K,2,3)')
    if bounds.ndim == 4 and bounds.shape[0] != len(tips):
        raise ValueError('dynamic key bounds timeline mismatch')
    if not bounds.shape[-3] or np.any(bounds[..., 1, :] < bounds[..., 0, :]):
        raise ValueError('invalid key bounds')
    if isinstance(tolerance_mm, bool) or not np.isfinite(tolerance_mm) or tolerance_mm < 0:
        raise ValueError('contact tolerance must be nonnegative finite millimeters')
    if np.any(key[~mask] != -1):
        raise ValueError('inactive reference keys must be -1')
    frame, finger = np.nonzero(mask)
    k = key[frame, finger]
    if np.any((k < 0) | (k >= bounds.shape[-3])):
        raise ValueError('invalid active reference key index')
    if not len(k):
        return {'status': 'not_measured', 'reason': 'no reference-assigned contacts', 'eligible_contacts': 0}
    b = bounds[k] if bounds.ndim == 3 else bounds[frame, k]
    active = tips[frame, finger]
    inside_xy = np.all((active[:, :2] >= b[:, 0, :2]) & (active[:, :2] <= b[:, 1, :2]), axis=1)
    signed_mm = (active[:, 2] - b[:, 1, 2]) * 1000
    hit = inside_xy & (np.abs(signed_mm) <= tolerance_mm)
    penetration = np.maximum(-signed_mm[inside_xy], 0)
    return {'status': 'measured', 'proxy': 'reference-assigned point / axis-aligned key box top',
            'key_bounds_mode': 'static' if bounds.ndim == 3 else 'per-frame',
            'eligible_contacts': int(len(k)), 'xy_inside_count': int(inside_xy.sum()),
            'surface_hit_count': int(hit.sum()), 'xy_inside_ratio': float(inside_xy.mean()),
            'surface_contact_ratio': float(hit.mean()), 'tolerance_mm': float(tolerance_mm),
            'absolute_vertical_error_mm': summary(np.abs(signed_mm), 'mm'),
            'target_key_point_penetration_mm': summary(penetration, 'mm'),
            'full_mesh_collision': {'status': 'not_measured', 'reason': 'box-point proxy cannot certify skin/key/self collision'}}

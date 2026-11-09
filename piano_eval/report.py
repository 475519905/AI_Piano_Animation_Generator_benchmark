"""Strict input contract, explicit missingness, and reproducible aggregation."""
import sys
import numpy as np
from . import __version__, metrics as m

PRED_KEYS = {'time_s', 'points_m', 'point_ids', 'rotations_xyzw', 'rotation_ids',
             'finger_labels', 'event_ids', 'event_start_s', 'fingertips_m'}
REF_KEYS = PRED_KEYS | {'contact_mask', 'contact_key', 'key_bounds_m'}
FRAME = 'keyboard_x_right_y_away_z_up'
FINGERS = 'right_thumb_to_little_then_left_thumb_to_little'


def missing(reason):
    return {'status': 'not_measured', 'reason': reason}


def ids_match(pred, ref, key, count):
    if key not in pred or key not in ref:
        raise ValueError(f'{key} required in both archives')
    a, b = np.asarray(pred[key]), np.asarray(ref[key])
    if a.dtype.kind != 'U' or b.dtype.kind != 'U' or a.shape != (count,) or b.shape != a.shape:
        raise ValueError(f'{key}: expected Unicode IDs with shape ({count},)')
    if len(set(a.tolist())) != count or any(not v for v in a.tolist()) or not np.array_equal(a, b):
        raise ValueError(f'{key}: unique nonempty IDs must match exactly, including order')
    return a.tolist()


def validate_metadata(meta):
    if not isinstance(meta, dict) or type(meta.get('schema_version')) is not int or meta.get('schema_version') != 1:
        raise ValueError('metadata schema_version must be 1')
    allowed = {'schema_version','sample_id','piece_group','performance_id','coordinate_calibration_id','postprocessing','position_unit','coordinate_frame','finger_order','split','stage','conditioning','sampling_unit','fps','contact_tolerance_mm'}
    if set(meta) - allowed:
        raise ValueError('unknown metadata fields; do not pass service configuration or credentials')
    for key in ('sample_id', 'piece_group', 'performance_id', 'coordinate_calibration_id', 'postprocessing'):
        if not isinstance(meta.get(key), str) or not meta[key].strip():
            raise ValueError(f'nonempty {key} required')
    if meta.get('position_unit') != 'm' or meta.get('coordinate_frame') != FRAME or meta.get('finger_order') != FINGERS:
        raise ValueError('units/frame/finger order must match the published canonical contract')
    if meta.get('split') not in ('train', 'validation', 'test'):
        raise ValueError('split must be train, validation or test')
    if meta.get('stage') not in ('raw_prediction', 'final_evaluated'):
        raise ValueError('stage must disclose raw_prediction or final_evaluated')
    if meta.get('conditioning') not in ('predicted_fingering', 'reference_fingering', 'not_applicable'):
        raise ValueError('conditioning must disclose whether reference fingering was supplied')
    if meta.get('sampling_unit') != 'full_performance':
        raise ValueError('v0.1 aggregation requires one full performance per sample')
    fps = meta.get('fps')
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not np.isfinite(fps) or fps <= 0:
        raise ValueError('fps must be a positive finite number')
    return float(fps)


def evaluate(pred, ref, meta):
    fps = validate_metadata(meta)
    if set(pred) - PRED_KEYS or set(ref) - REF_KEYS:
        raise ValueError('unknown array keys; check DATA_SPEC.md for spelling and role')
    if 'time_s' not in pred or 'time_s' not in ref:
        raise ValueError('both archives require time_s')
    t, rt = m.numeric(pred['time_s'], 'prediction time_s', 1), m.numeric(ref['time_s'], 'reference time_s', 1)
    if not len(t) or t.shape != rt.shape or not np.allclose(t, rt, rtol=0, atol=1e-9):
        raise ValueError('timelines must match exactly; no hidden shift/interpolation')
    if len(t) > 1 and not np.allclose(np.diff(t), 1 / fps, rtol=1e-6, atol=1e-9):
        raise ValueError('timestamps must be strictly increasing with declared uniform fps')
    for arrays in (pred, ref):
        for key in ('points_m', 'rotations_xyzw', 'fingertips_m', 'contact_mask', 'contact_key'):
            if key in arrays and (np.asarray(arrays[key]).ndim == 0 or len(arrays[key]) != len(t)):
                raise ValueError(f'{key}: timeline length mismatch')
    metrics = {}; layout = {}
    for field, ids, fn, name in [('points_m', 'point_ids', m.point_errors, 'pose'),
                               ('rotations_xyzw', 'rotation_ids', m.rotation_errors, 'rotation')]:
        if field in pred and field in ref:
            if np.asarray(pred[field]).ndim != 3:
                raise ValueError(f'{field}: expected three dimensions')
            layout[ids] = ids_match(pred, ref, ids, np.asarray(pred[field]).shape[1])
            metrics[name] = fn(pred[field], ref[field])
        else:
            metrics[name] = missing(f'{field} reference/prediction pair absent')
    for key, fn, label in [('finger_labels', m.fingering, 'fingering'), ('event_start_s', m.timing_errors, 'onset_timing')]:
        if key in pred and key in ref:
            ids_match(pred, ref, 'event_ids', len(np.asarray(pred[key])))
            metrics[label] = fn(pred[key], ref[key])
        else:
            metrics[label] = missing(f'{key} reference/prediction pair absent')
    if 'fingertips_m' in pred:
        tips = m.numeric(pred['fingertips_m'], 'fingertips', 3)
        if tips.shape[1:] != (10, 3):
            raise ValueError('fingertips must have shape (T,10,3)')
        metrics['temporal'] = m.temporal_motion(tips, fps, ref.get('contact_mask'))
        fields = ('contact_mask', 'contact_key', 'key_bounds_m')
        if all(k in ref for k in fields):
            if 'contact_tolerance_mm' not in meta:
                raise ValueError('explicit contact_tolerance_mm required; no universal default')
            metrics['assigned_contact'] = m.assigned_contact(tips, *(ref[k] for k in fields), meta['contact_tolerance_mm'])
        else:
            metrics['assigned_contact'] = missing('independent reference contact/key geometry absent')
    else:
        metrics['temporal'] = missing('no evaluated fingertip trajectory')
        metrics['assigned_contact'] = missing('no evaluated fingertip trajectory')
    metrics['full_mesh_collision'] = missing('not implemented; requires an independent evaluated-mesh audit')
    metrics['human_plausibility'] = missing('requires preregistered blinded review; not inferred from numeric proxies')
    return {'schema_version': 1, 'tool_version': __version__, 'metadata': dict(meta),
            'runtime': {'python': sys.version.split()[0], 'numpy': np.__version__},
            'frames': len(t), 'layout': layout, 'metrics': metrics,
            'overall_pass': None, 'overall_score': None,
            'warning': 'No universal pass threshold; a synthetic run is not a real-model benchmark.'}


def check_splits(rows, hold_out_performers=False):
    if not isinstance(rows, list) or not rows:
        raise ValueError('split manifest must be a nonempty list')
    seen = set(); groups = {}; keys = ['piece_group', 'performance_id']
    if hold_out_performers:
        keys.append('performer_id')
    for row in rows:
        if not isinstance(row, dict) or row.get('split') not in ('train', 'validation', 'test'):
            raise ValueError('invalid split record')
        sid = row.get('sample_id')
        if not isinstance(sid, str) or not sid or sid in seen:
            raise ValueError('sample IDs must be unique nonempty strings')
        seen.add(sid)
        for key in keys:
            value = row.get(key)
            if not isinstance(value, str) or not value:
                raise ValueError(f'nonempty {key} required')
            previous = groups.setdefault((key, value), row['split'])
            if previous != row['split']:
                raise ValueError(f'split leakage declared for {key}: {value}')
    return {'status': 'passed_declaration_check', 'samples': len(rows), 'group_keys': keys,
            'proves_data_independence': False,
            'limitation': 'Does not fingerprint recordings, detect mislabeled groups, or prove a test set was never viewed.'}


def aggregate(reports):
    if not reports:
        raise ValueError('no reports')
    seen = set(); signatures = set(); results = {'performances': len(reports), 'metrics': {}}
    for r in reports:
        if r.get('schema_version') != 1 or r.get('tool_version') != __version__:
            raise ValueError('report schema/tool version mismatch')
        meta = r['metadata']; validate_metadata(meta)
        if meta['performance_id'] in seen:
            raise ValueError('one report per performance; do not macro-average arbitrary clip fragments')
        seen.add(meta['performance_id'])
        signatures.add(tuple(meta.get(k) for k in ('fps', 'stage', 'conditioning', 'split', 'postprocessing', 'contact_tolerance_mm')))
    if len(signatures) != 1:
        raise ValueError('fps/stage/conditioning/split/postprocessing/tolerance differ; report separately')
    for key in ('pose', 'fingering', 'assigned_contact', 'onset_timing'):
        selected = [r for r in reports if r['metrics'][key]['status'] == 'measured']
        rows = [r['metrics'][key] for r in selected]
        result = {'measured_performances': len(rows), 'missing_performances': len(reports)-len(rows)}
        if not rows:
            result.update(status='not_measured'); results['metrics'][key] = result; continue
        result['status'] = 'measured'
        if key == 'pose':
            if len({tuple(r['layout']['point_ids']) for r in selected}) != 1:
                raise ValueError('point schemas differ; do not pool incomparable marker sets')
            result.update(micro_coordinate_rmse_mm=float(np.sqrt(sum(v['squared_coordinate_error_sum_m2'] for v in rows)/sum(v['coordinate_count'] for v in rows))*1000),
                          micro_mean_point_error_mm=sum(v['point_distance_sum_m'] for v in rows)/sum(v['point_count'] for v in rows)*1000,
                          macro_mean_point_error_mm=float(np.mean([v['mean_point_error_mm'] for v in rows])),
                          macro_coordinate_rmse_mm=float(np.mean([v['coordinate_rmse_mm'] for v in rows])))
        elif key == 'fingering':
            total = sum(v['events'] for v in rows)
            result.update(events=total, micro_exact_accuracy=sum(v['exact_correct'] for v in rows)/total,
                          micro_hand_accuracy=sum(v['hand_correct'] for v in rows)/total,
                          macro_exact_accuracy=float(np.mean([v['exact_accuracy'] for v in rows])))
        elif key == 'assigned_contact':
            total = sum(v['eligible_contacts'] for v in rows)
            result.update(eligible_contacts=total, micro_surface_contact_ratio=sum(v['surface_hit_count'] for v in rows)/total,
                          macro_surface_contact_ratio=float(np.mean([v['surface_contact_ratio'] for v in rows])))
        else:
            result.update(events=sum(v['count'] for v in rows), micro_mae_ms=sum(v['absolute_error_sum_ms'] for v in rows)/sum(v['count'] for v in rows),
                          macro_mae_ms=float(np.mean([v['mean'] for v in rows])))
        results['metrics'][key] = result
    results['overall_pass'] = None
    results['warning'] = 'Micro statistics pool counts/sums. Macro weights full performances equally. Percentiles are not averaged.'
    return results

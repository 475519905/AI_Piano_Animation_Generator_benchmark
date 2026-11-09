from pathlib import Path
import copy,json,tempfile,unittest
import numpy as np
from piano_eval import metrics as m
from piano_eval.report import evaluate,aggregate,check_splits
from piano_eval.__main__ import main as cli
from examples.generate_synthetic import fixture

class MetricsTests(unittest.TestCase):
    def test_coordinate_rmse_is_not_mean_point_distance(self):
        x=m.point_errors(np.ones((2,1,3))*.001,np.zeros((2,1,3)))
        self.assertAlmostEqual(x['coordinate_rmse_mm'],1);self.assertAlmostEqual(x['mean_point_error_mm'],np.sqrt(3))
    def test_point_does_not_align_translation_away(self):
        p=np.zeros((1,2,3));p[...,0]=.01
        self.assertAlmostEqual(m.point_errors(p,np.zeros_like(p))['mean_point_error_mm'],10)
    def test_nonfinite_rejected(self):
        p=np.zeros((1,1,3));p[0,0,0]=np.nan
        with self.assertRaises(ValueError):m.point_errors(p,np.zeros_like(p))
    def test_empty_points_rejected(self):
        with self.assertRaises(ValueError):m.point_errors(np.empty((0,1,3)),np.empty((0,1,3)))
    def test_shape_mismatch_rejected(self):
        with self.assertRaises(ValueError):m.point_errors(np.zeros((2,1,3)),np.zeros((1,1,3)))
    def test_quaternion_double_cover(self):
        q=np.array([[[0.,0.,0.,1.]]]);self.assertAlmostEqual(m.rotation_errors(q,-q)['mean'],0)
    def test_quaternion_180_degrees(self):
        self.assertAlmostEqual(m.rotation_errors([[[1,0,0,0]]],[[[0,0,0,1]]])['mean'],180)
    def test_quaternion_inputs_not_modified(self):
        a=np.array([[[0.,0.,0.,2.]]]);b=a.copy();m.rotation_errors(a,b);np.testing.assert_array_equal(a,[[[0,0,0,2]]]);np.testing.assert_array_equal(a,b)
    def test_zero_quaternion_rejected(self):
        with self.assertRaises(ValueError):m.rotation_errors(np.zeros((1,1,4)),np.ones((1,1,4)))
    def test_abstentions_penalized(self):
        r=m.fingering(np.array([0,-1,7]),np.array([0,6,8]));self.assertEqual(r['exact_accuracy'],1/3);self.assertEqual(r['hand_accuracy'],2/3);self.assertEqual(r['prediction_coverage'],2/3)
    def test_empty_fingering_not_perfect(self):self.assertEqual(m.fingering(np.array([],int),np.array([],int))['status'],'not_measured')
    def test_label_bounds(self):
        with self.assertRaises(ValueError):m.fingering(np.array([10]),np.array([0]))
    def test_labels_must_be_integers(self):
        with self.assertRaises(ValueError):m.fingering(np.array([0.5]),np.array([0]))
    def test_timing_sign_and_absolute(self):
        r=m.timing_errors([.02,.99],[0,1]);self.assertAlmostEqual(r['mean'],15);self.assertAlmostEqual(r['signed_mean_ms'],5)
    def test_empty_summary_not_zero(self):self.assertEqual(m.summary([], 'mm')['status'],'not_measured')
    def test_linear_speed_uses_seconds(self):
        for fps in (30,60):
            t=np.arange(50)/fps;p=np.zeros((50,1,3));p[:,0,0]=t*.01
            self.assertAlmostEqual(m.temporal_motion(p,fps)['speed']['mean'],10)
    def test_cubic_jerk_units_across_frame_rates(self):
        for fps in (30,60):
            t=np.arange(50)/fps;p=np.zeros((50,1,3));p[:,0,0]=.001*t**3
            self.assertAlmostEqual(m.temporal_motion(p,fps)['jerk']['mean'],6,places=7)
    def test_short_held_sequence_is_missing(self):
        for n in (1,2,3):self.assertEqual(m.temporal_motion(np.zeros((n,10,3)),30,np.ones((n,10),bool))['held_jerk']['status'],'not_measured')
    def test_held_jerk_requires_four_consecutive_contacts(self):
        mask=np.zeros((8,1),bool);mask[[1,3,5,7]]=True
        self.assertEqual(m.temporal_motion(np.zeros((8,1,3)),30,mask)['held_jerk']['status'],'not_measured')
    def test_invalid_fps(self):
        with self.assertRaises(ValueError):m.temporal_motion(np.zeros((5,1,3)),0)

class ReportTests(unittest.TestCase):
    def setUp(self):self.p,self.r,self.meta=fixture()
    def evaluate(self):return evaluate(self.p,self.r,self.meta)
    def test_identical_fixture_not_claimed_real_model_score(self):
        out=self.evaluate();self.assertEqual(out['metrics']['pose']['coordinate_rmse_mm'],0);self.assertEqual(out['metrics']['assigned_contact']['surface_contact_ratio'],1);self.assertIsNone(out['overall_pass']);self.assertEqual(out['metrics']['full_mesh_collision']['status'],'not_measured')
    def test_contact_checks_y_not_just_key_x(self):
        self.p['fingertips_m'][20:60,0,1]=.2
        self.assertEqual(self.evaluate()['metrics']['assigned_contact']['surface_contact_ratio'],0)
    def test_floating_negative_control(self):
        self.p['fingertips_m'][...,2]+=.01;self.assertEqual(self.evaluate()['metrics']['assigned_contact']['surface_contact_ratio'],0)
    def test_penetration_negative_control(self):
        self.p['fingertips_m'][20:60,0,2]=-.01;x=self.evaluate()['metrics']['assigned_contact'];self.assertEqual(x['surface_contact_ratio'],0);self.assertAlmostEqual(x['target_key_point_penetration_mm']['max'],10)
    def test_no_contact_reference_is_not_zero_penetration(self):
        for key in ('contact_mask','contact_key','key_bounds_m'):self.r.pop(key)
        self.assertEqual(self.evaluate()['metrics']['assigned_contact']['status'],'not_measured')
    def test_empty_contact_mask_is_not_perfect(self):
        self.r['contact_mask'][:]=False;self.r['contact_key'][:]=-1
        self.assertEqual(self.evaluate()['metrics']['assigned_contact']['status'],'not_measured')
    def test_bad_key_rejected(self):
        self.r['contact_key'][20,0]=99
        with self.assertRaises(ValueError):self.evaluate()
    def test_nonboolean_mask_rejected(self):
        self.r['contact_mask']=self.r['contact_mask'].astype(float)
        with self.assertRaises(ValueError):self.evaluate()
    def test_dynamic_key_height_used(self):
        b=np.repeat(self.r['key_bounds_m'][None],len(self.r['time_s']),axis=0);b[20:60,:,1,2]-=.01;self.r['key_bounds_m']=b
        self.assertEqual(self.evaluate()['metrics']['assigned_contact']['surface_contact_ratio'],0)
    def test_invalid_tolerance(self):
        self.meta['contact_tolerance_mm']=-1
        with self.assertRaises(ValueError):self.evaluate()
    def test_no_default_contact_tolerance(self):
        self.meta.pop('contact_tolerance_mm')
        with self.assertRaises(ValueError):self.evaluate()
    def test_no_hidden_time_shift(self):
        self.p['time_s']+=.01
        with self.assertRaises(ValueError):self.evaluate()
    def test_bad_fps_rejected(self):
        self.meta['fps']=60
        with self.assertRaises(ValueError):self.evaluate()
    def test_wrong_units_rejected(self):
        self.meta['position_unit']='cm'
        with self.assertRaises(ValueError):self.evaluate()
    def test_teacher_forcing_must_be_declared(self):
        self.meta.pop('conditioning')
        with self.assertRaises(ValueError):self.evaluate()
    def test_marker_identity_mismatch(self):
        self.p['point_ids']=self.p['point_ids'][::-1]
        with self.assertRaises(ValueError):self.evaluate()
    def test_event_identity_mismatch(self):
        self.p['event_ids']=np.array(['different-event'])
        with self.assertRaises(ValueError):self.evaluate()
    def test_unknown_field_rejected(self):
        self.p['marker_poistions']=np.zeros(1)
        with self.assertRaises(ValueError):self.evaluate()
    def test_nan_contact_tip_rejected(self):
        self.p['fingertips_m'][0,0,0]=np.nan
        with self.assertRaises(ValueError):self.evaluate()
    def test_stationary_trajectory_is_only_descriptive(self):
        self.p['fingertips_m'][:]=self.p['fingertips_m'][0]
        out=self.evaluate();self.assertEqual(out['metrics']['temporal']['jerk']['mean'],0);self.assertIsNone(out['overall_score']);self.assertEqual(out['metrics']['assigned_contact']['surface_contact_ratio'],0)
    def test_raw_and_final_cannot_be_pooled(self):
        a=self.evaluate();self.meta['performance_id']='other';self.meta['stage']='raw_prediction';b=self.evaluate()
        with self.assertRaises(ValueError):aggregate([a,b])
    def test_duplicate_performance_rejected(self):
        a=self.evaluate()
        with self.assertRaises(ValueError):aggregate([a,a])
    def test_missing_modalities_reported(self):
        out=evaluate({'time_s':self.p['time_s']},{'time_s':self.r['time_s']},self.meta)
        self.assertEqual(out['metrics']['pose']['status'],'not_measured');self.assertEqual(aggregate([out])['metrics']['pose']['missing_performances'],1)
    def test_micro_rmse_pools_squared_errors_not_rmse(self):
        a=self.evaluate();self.meta['performance_id']='other';self.meta['sample_id']='other';self.p['points_m']+=.002;b=self.evaluate();v=aggregate([a,b])['metrics']['pose']
        self.assertAlmostEqual(v['micro_coordinate_rmse_mm'],np.sqrt(2));self.assertAlmostEqual(v['macro_coordinate_rmse_mm'],1)
    def test_empty_aggregate_rejected(self):
        with self.assertRaises(ValueError):aggregate([])
    def test_cli_npz_roundtrip_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);np.savez(p/'p.npz',**self.p);np.savez(p/'r.npz',**self.r);(p/'m.json').write_text(json.dumps(self.meta))
            args=['evaluate','--prediction',str(p/'p.npz'),'--reference',str(p/'r.npz'),'--metadata',str(p/'m.json'),'--output',str(p/'out.json')]
            self.assertEqual(cli(args),0);data=json.loads((p/'out.json').read_text());self.assertIn('input_sha256',data)
            with self.assertRaises(SystemExit):cli(args)
    def test_metadata_rejects_service_fields(self):
        self.meta['api_key']='not-a-real-key'
        with self.assertRaises(ValueError):self.evaluate()
    def test_boolean_schema_version_rejected(self):
        self.meta['schema_version']=True
        with self.assertRaises(ValueError):self.evaluate()
    def test_mixed_conditioning_not_pooled(self):
        a=self.evaluate();self.meta['performance_id']='other';self.meta['conditioning']='reference_fingering';b=self.evaluate()
        with self.assertRaises(ValueError):aggregate([a,b])
    def test_contact_point_below_but_outside_xy_is_not_collision_proof(self):
        self.p['fingertips_m'][20:60,0,0]=1;self.p['fingertips_m'][20:60,0,2]=-.1
        x=self.evaluate()['metrics']['assigned_contact'];self.assertEqual(x['surface_contact_ratio'],0)
        self.assertEqual(x['target_key_point_penetration_mm']['status'],'not_measured')
    def test_pickle_array_rejected(self):
        from piano_eval.__main__ import load_npz
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'bad.npz';np.savez(p,time_s=np.array([object()],dtype=object))
            with self.assertRaises(ValueError):load_npz(p)

class SplitTests(unittest.TestCase):
    def rows(self):return [dict(sample_id='a',piece_group='piece-a',performance_id='perf-a',performer_id='person-a',split='train'),dict(sample_id='b',piece_group='piece-b',performance_id='perf-b',performer_id='person-a',split='test')]
    def test_declared_disjoint_groups_pass_with_limited_claim(self):
        r=check_splits(self.rows());self.assertFalse(r['proves_data_independence'])
    def test_piece_variants_cannot_cross_splits(self):
        r=self.rows();r[1]['piece_group']=r[0]['piece_group']
        with self.assertRaises(ValueError):check_splits(r)
    def test_performance_clips_cannot_cross_splits(self):
        r=self.rows();r[1]['performance_id']=r[0]['performance_id']
        with self.assertRaises(ValueError):check_splits(r)
    def test_performer_holdout_is_optional_and_explicit(self):
        with self.assertRaises(ValueError):check_splits(self.rows(),hold_out_performers=True)
    def test_missing_group_rejected(self):
        r=self.rows();r[0].pop('piece_group')
        with self.assertRaises(ValueError):check_splits(r)
    def test_empty_manifest_rejected(self):
        with self.assertRaises(ValueError):check_splits([])
if __name__=='__main__':unittest.main()

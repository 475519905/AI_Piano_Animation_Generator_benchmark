"""Generate arithmetic fixtures only: no music, recorded motion, assets or model."""
from pathlib import Path
import argparse,json
import numpy as np


def fixture(fps=30, frames=90):
    t = np.arange(frames, dtype=np.float64) / fps
    tip = np.zeros((frames,10,3), dtype=np.float64)
    tip[:,:,0] = np.arange(10)[None,:] * .025
    tip[:,:,1] = .05; tip[:,:,2] = .03
    tip[:20,0,2] = np.linspace(.03,0,20)
    tip[20:60,0,2] = 0
    tip[60:,0,2] = np.linspace(0,.03,frames-60)
    mask = np.zeros((frames,10),bool);mask[20:60,0] = True
    key = np.full((frames,10),-1,np.int64);key[mask]=0
    bounds = np.array([[[-.01,0,-.01],[.01,.1,0]],[[.015,0,-.01],[.035,.1,0]]])
    q = np.zeros((frames,1,4));q[:,:,3]=1
    shared = dict(time_s=t,points_m=tip.copy(),point_ids=np.array([f'finger_tip_{i}' for i in range(10)]),rotations_xyzw=q,
                  rotation_ids=np.array(['synthetic_body']),finger_labels=np.array([0],np.int64),event_ids=np.array(['synthetic-event-0']),event_start_s=np.array([20/fps]))
    reference = {**shared,'contact_mask':mask,'contact_key':key,'key_bounds_m':bounds}
    prediction = {**{k:v.copy() for k,v in shared.items()},'fingertips_m':tip.copy()}
    metadata = dict(schema_version=1,sample_id='synthetic-001',piece_group='invented-pattern-A',performance_id='invented-performance-A',
                    coordinate_calibration_id='synthetic-keyboard-v1',postprocessing='none',position_unit='m',
                    coordinate_frame='keyboard_x_right_y_away_z_up',finger_order='right_thumb_to_little_then_left_thumb_to_little',
                    split='test',stage='final_evaluated',conditioning='predicted_fingering',sampling_unit='full_performance',fps=fps,contact_tolerance_mm=2.0)
    return prediction,reference,metadata


def main():
    p=argparse.ArgumentParser();p.add_argument('output_directory');a=p.parse_args();out=Path(a.output_directory)
    if out.exists():p.error('destination exists; refusing overwrite')
    out.mkdir(parents=True);pred,ref,meta=fixture()
    np.savez_compressed(out/'reference.npz',**ref)
    (out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    variants={'identical':pred,'floating':{k:v.copy() for k,v in pred.items()},'jitter':{k:v.copy() for k,v in pred.items()},'wrong_hand':{k:v.copy() for k,v in pred.items()}}
    for k in ('points_m','fingertips_m'):
        variants['floating'][k][...,2] += .010
        variants['jitter'][k][...,2] += ((-1.)**np.arange(len(pred['time_s'])))[:,None]*.004
    variants['wrong_hand']['finger_labels'][:]=5
    for name,data in variants.items():np.savez_compressed(out/(name+'.npz'),**data)
    print(out)

if __name__=='__main__':main()

# Blender / C4D adapter checklist

This repository **does not ship a validated DCC exporter**. A producer must supply the canonical arrays.
Do not claim host integration based only on the synthetic examples.

1. Freeze scene, rig, keyboard geometry, unit scale and coordinate transforms. Give the transform a calibration ID.
2. Iterate the declared frame times, including the same start-time convention as reference MIDI.
   Evaluate animation/deformers/constraints through the host's supported evaluation process before reading positions.
3. Export evaluated fingertip landmarks, not just controller targets. Export named joint/marker positions and rotations
   with explicit common ID order. Record whether positions represent bones, skin points or artist-defined markers.
4. Transform both reference and prediction into the fixed keyboard frame in **meters**. Do not fit the transform to each predicted clip.
   Convert quaternions with a documented axis convention; export xyzw.
5. For the contact proxy, independently derive expected finger/key contacts from the reference annotation.
   Export actual per-frame key bounds if keys move. Distinguish MIDI note duration from physical finger contact and pedal sustain;
   they are not automatically the same timeline. Declare the annotation policy.
6. Reference-assigned contact is conditional on the annotation's fingering. Alternative valid finger choices require
   a separate note/key-event matching evaluation, which v0.1 does not implement.
7. Supply NPZ numeric/Unicode arrays only, and a separate metadata JSON. Avoid pickle, host objects and scene assets.
8. Validate a known 1 cm translation, 180 degree rotation, 1-frame delay, raised fingertip and deformed rig.
   Check expected errors before running any real benchmark.
9. Compare raw and final evaluated motion. Do not measure an IK target and render a different skin result.
10. A point-to-key-box metric does not certify mesh collision or anatomy. Implement/report those separately, with missingness explicit.

Adapters must not silently repair missing tracks or reference labels. If the host cannot evaluate a frame, record an invalid run,
not a substituted identity pose. No model, proprietary plugin or DCC binary is required by the scorer itself.

# Publication scope and provenance

This is a fresh, independent evaluator implementation, created for public distribution on 2026-11-09.
It contains mathematical measurement functions, explicit input validation, methodology documents and authored synthetic tests.
It is **not** a copy of the private AiPiano repository, its commit history, models or DCC assets.

Published:
- reference position/rotation/fingering/timing comparisons;
- a narrowly labeled reference-assigned point-to-key-box proxy;
- time-normalized motion derivatives, explicit missingness and aggregation rules;
- split-declaration checks, negative controls, synthetic data generator and tests.

Excluded:
- inference/training/model code and weights;
- real songs, MIDI libraries, recordings, original performance tracks and private annotations;
- characters, rigs, piano meshes/materials, renders and commercial deliverables;
- service endpoints, keys, accounts, internal network addresses and deployment configuration;
- proprietary/reverse-engineering research files and third-party implementation source.

Private training/deployment repositories remain private. The MIT license covers the newly authored files here;
it does not license excluded data, models or third-party assets. NumPy remains under its upstream license.

The protocol documents lessons from internal evaluation, but v0.1 is not a bit-for-bit implementation of every historical score.
In particular, it distinguishes coordinate RMSE from mean point distance, uses seconds in derivative units,
requires explicit missingness, and does not publish historical empirical gate values as universal quality standards.

No real model was benchmarked for this release. Passing the tests means the covered arithmetic/validation cases behaved as expected,
not that a pianist animation model is accurate, natural, anatomically safe or ready for production.

Background references (no code or datasets copied):
- [Human3.6M evaluation instructions](https://vision.imar.ro/human3.6m/readme_challenge.php): general MPJPE terminology.
- [PianoMotion10M](https://arxiv.org/abs/2406.09326): related piano motion benchmark research; this repository does not claim to reproduce its protocol.

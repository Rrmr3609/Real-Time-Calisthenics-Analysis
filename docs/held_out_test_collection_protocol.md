# Held-Out Test Collection Protocol

## Purpose

This document defines the held-out test set before any final-test recordings
are analysed by the baseline or enhanced systems.

Development-stage scientific decisions were frozen at Git commit `40a204c`.
The held-out test set must not be used to alter thresholds, temporal rules,
class priority, smoothing, side selection, event tolerance, or any other
scientific configuration.

## Test-set structure

The held-out test set consists of four newly recorded self-recorded clips.

### Collection deviation

The original pre-registered collection protocol specified six clips. Before
ground-truth annotation and before either system was run on the held-out data,
the recording plan was reduced for feasibility.

Four clips were ultimately collected:

| Clip ID | Camera view | Recorded collection condition |
| --- | --- | --- |
| `test01` | side | Mixed normal and reduced-depth attempts |
| `test02` | side | Mixed normal, incomplete-extension and alignment-deviation attempts |
| `test03` | side | Alignment-deviation-focused attempts |
| `test04` | side-diagonal | Mixed-quality attempts covering the planned form conditions |

The recording descriptions above document collection intent only. They are not
ground-truth labels. Ground truth must be assigned independently from visible
source-video evidence using the frozen manual annotation protocol.

The reduction and change in clip composition occurred before viewing baseline
or enhanced predictions for these recordings. No held-out clip was selected,
discarded, repeated, or retained based on system performance.

## Recording conditions

Recordings should remain within the intended operating scope of the project:

- one visible participant;
- controlled indoor environment;
- fixed camera;
- full push-up movement visible throughout the usable clip;
- side or side-diagonal view as specified above;
- no intentional camera movement during a repetition;
- ordinary consumer-camera recording without post-processing that changes
  movement timing.

The participant should begin in a stable top position and leave sufficient
video before the first attempt and after the final attempt for visual review.

A clip should only be re-recorded because of an obvious acquisition failure
such as severe occlusion, camera movement, corrupted video, or the participant
leaving the frame. It must not be re-recorded because of how either system
would score it.

## Independence from development tuning

After recording:

1. add source-video metadata and hashes to
   `data/manifests/test_dataset_manifest.csv`;
2. assign all rows to split `test`;
3. annotate repetitions manually from source video only;
4. do not inspect baseline or enhanced predictions during annotation;
5. review and freeze the test annotation CSV;
6. only after annotation freeze, run the frozen baseline and enhanced systems;
7. evaluate both methods using the frozen `0.50`-second event tolerance;
8. report final-test results without retuning.

The recording script, intended error type, development predictions, development
feature values, filenames, and system outputs must not be used as substitutes
for visual ground-truth judgement.

## Ground-truth rules

Use the existing `docs/manual_annotation_protocol.md` unchanged.

In particular:

- an evaluable attempt is identified visually rather than from system
  thresholds;
- completion is the first frame after ascent at which the attempt has visibly
  returned to its ending/top posture;
- deviation flags are assigned from visible movement;
- ambiguous or non-evaluable fragments remain explicitly represented;
- the canonical class follows the already frozen annotation rule and priority;
- predictions must remain hidden until test annotations are frozen.

## Final-test lock

The scientific configuration entering this test is the configuration frozen
at commit `40a204c`, including:

- enhanced depth threshold: `65.0` degrees;
- enhanced extension threshold: `150.0` degrees;
- enhanced alignment threshold: `160.0` degrees;
- formal event tolerance: `0.50` seconds;
- unchanged baseline configuration.

A poor held-out result is a result to report, not a reason to reopen
development tuning.

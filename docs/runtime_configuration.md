# Recorded-run configuration and provenance

## Authority and precedence

`configs/default.yaml` is the primary configuration source for both
recorded-video runners. The runners require every documented field and reject
unknown fields, malformed YAML, invalid types and out-of-range values. They do
not supply fallback values when a field is absent.

Configuration precedence is:

1. values loaded from the file supplied by `--config`;
2. an explicit supported command-line override.

The only current numeric override is enhanced-runner `--alpha`, which replaces
`features.ema_alpha`. If it is present, the resolved value and the mapping
`features.ema_alpha` are recorded under
`configuration.explicit_cli_overrides` in run metadata. If it is absent, the
YAML value is used and the override mapping is empty.

`--split` and `--run-id` describe run identity rather than analysis
configuration. `--split` is mandatory and accepts only `development` or
`test`. `--run-id` defaults to `--clip-id`.

## Configuration fields

All angles are in degrees and all frame counts are integers.

### `config_schema_version`

Required schema version. The current accepted value is `1`.

### `pose`

| Field | Default | Validation and use |
| --- | ---: | --- |
| `minimum_detection_confidence` | `0.5` | Greater than 0 and at most 1; passed to MediaPipe Pose |
| `minimum_tracking_confidence` | `0.5` | Greater than 0 and at most 1; passed to MediaPipe Pose |

### `baseline`

These values preserve the intentionally simple raw-threshold comparator.

| Field | Default | Validation and use |
| --- | ---: | --- |
| `top_elbow_angle` | `150.0` | 0 to 180; sticky baseline top threshold |
| `bottom_elbow_angle` | `100.0` | 0 to 180 and lower than the top threshold; sticky baseline bottom threshold |
| `top_extension_warning_threshold` | `150.0` | 0 to 180; diagnostic-only frame warning |
| `depth_warning_threshold` | `100.0` | 0 to 180; diagnostic-only frame warning |
| `alignment_warning_minimum` | `160.0` | 0 to 180; diagnostic-only frame warning |

### `features`

| Field | Default | Validation and use |
| --- | ---: | --- |
| `minimum_landmark_visibility` | `0.5` | 0 to 1; required MediaPipe landmark visibility |
| `side_acquisition_frames` | `3` | At least 1; consecutive frames required to acquire an elbow side |
| `side_switch_frames` | `5` | At least 1; consecutive frames required to switch sides |
| `side_switch_margin` | `0.10` | 0 to 1; visibility-score advantage required for switching |
| `missing_side_grace_frames` | `5` | At least 0; retained-side grace after missing evidence |
| `ema_alpha` | `0.3` | Greater than 0 and at most 1; elbow and alignment EMA smoothing |

### `segmentation`

| Field | Default | Validation and use |
| --- | ---: | --- |
| `top_region_threshold` | `130.0` | 0 to 180; enhanced top-region threshold |
| `bottom_region_threshold` | `120.0` | 0 to 180 and lower than the top threshold; enhanced bottom-region threshold |
| `hysteresis` | `5.0` | 0 to 180; enhanced transition hysteresis |
| `phase_confirmation_frames` | `3` | At least 1; consecutive valid frames required for a phase transition |
| `missing_angle_grace_frames` | `5` | At least 0; missing-elbow tolerance |
| `minimum_repetition_frames` | `8` | At least 1; minimum inclusive completed-repetition duration |

### `classification`

| Field | Default | Validation and use |
| --- | ---: | --- |
| `depth_threshold` | `100.0` | 0 to 180; insufficient-depth rule |
| `extension_threshold` | `150.0` | 0 to 180; incomplete-extension rule |
| `alignment_minimum` | `160.0` | 0 to 180; alignment-deviation frame threshold |
| `alignment_deviation_min_frames` | `3` | At least 1; required deviating valid-alignment frames |
| `alignment_deviation_min_ratio` | `0.20` | 0 to 1; required deviating share of valid-alignment frames |
| `minimum_alignment_valid_ratio` | `0.50` | 0 to 1; required alignment coverage for an alignment decision |

These are provisional operational project values for the controlled
recordings. They are not universal or medical definitions. This configuration
milestone did not change any value, temporal rule, side-selector behavior or
classifier priority.

## Enhanced repetition measurement window

Enhanced repetition detection and within-repetition aggregation use one closed,
inclusive frame interval. The interval begins at the frame containing the
maximum genuine top observation (`elbow_angle >= top_region_threshold`)
available before the contiguous descent-confirmation sequence. That anchor is
frozen when the candidate sequence begins. The interval ends on the final
confirmation frame that completes the return to the top region.

Within this interval:

- `duration_frames` is `end_frame - start_frame + 1`;
- only the contiguous descent-candidate frames that cause the transition are
  retained, while an interrupted candidate sequence is discarded;
- tolerated missing-elbow frames after descent confirmation remain inside the
  interval;
- minimum elbow angle and bottom frame use every valid elbow observation,
  including the retained confirmation candidates;
- `start_top_angle` comes from the genuine top anchor;
- body-alignment observations use the same start and end frames without
  fabricating values for missing observations;
- alignment coverage is valid alignment observations divided by
  `duration_frames`, so complete availability produces coverage `1.0`.

The incomplete-extension feature is finalised causally after detection.
`end_top_angle` is the maximum valid elbow angle from the confirmed
return-to-top sequence and the continuing stable returned `top` phase. The
phase ends when the state machine confirms the next descent or otherwise
leaves `top`; a pending final repetition is flushed when the stream ends.
`top_extension_angle` uses this return peak only, because extension before the
descent does not describe the posture achieved after the repetition bottom.
This post-completion observation does not alter `end_frame`, repetition count,
duration, minimum elbow angle, alignment observations or alignment-coverage
denominator. It uses the existing stable phase boundary, so no new configured
observation length or tuned threshold is introduced.

Hysteresis, consecutive-frame confirmation and missing-frame tolerance control
the transitions. Attempts abandoned before completion do not contribute
measurements to a later repetition.

## Output identity and metadata

Each recorded run writes a shared `run_id` into every CSV row and uses it in
all output filenames. Before processing, the runner checks the complete CSV
and metadata set. Existing files cause a clear failure unless `--overwrite`
was explicitly supplied; overwrite removes the complete old set before new
writers are opened.

The baseline writes:

- `experiments/logs/<run-id>_baseline.csv`;
- `experiments/logs/<run-id>_baseline_metadata.json`.

The enhanced method writes:

- `experiments/logs/<run-id>_enhanced_temporal.csv`;
- `experiments/outputs/<run-id>_enhanced_repetitions.csv`;
- `experiments/outputs/<run-id>_enhanced_metadata.json`.

Enhanced temporal-CSV completion fields remain attached to the original
return-confirmation frame and therefore retain the confirmation-time
`completed_end_top_angle`. The repetition CSV is written after causal
return-top finalisation and records the representative return peak in
`end_top_angle` and `top_extension_angle`. The shared `rep_id` and unchanged
`end_frame` preserve the link between those records. Temporal-CSV
classification fields occur on the frame where that causal classification is
finalised, which can be later than the detector-completion row.

Metadata records run and clip identity, method, split, input and config paths,
SHA256 hashes, file size, source properties, the fully resolved configuration,
explicit overrides, installed core-library versions, Git
commit/branch/dirty state, UTC timestamps, timing-boundary definitions and
output paths. Paths inside the repository are stored project-relative to avoid
embedding a local username or checkout location. An input outside the
repository retains its resolved absolute path because no project-relative
identity exists.

Metadata is written atomically after resource cleanup. A successful run has
`status: completed`. A setup, processing or cleanup exception has
`status: failed`, an error type/message and no completion timestamp. A
user-requested early stop is recorded separately in the processing summary and
does not claim that the full clip was processed.

## Reproduction commands

From the repository root in PowerShell:

```powershell
& ".\.venv\Scripts\python.exe" src\run_video.py `
  --video "data\raw\development\example.mp4" `
  --clip-id "example" `
  --split development `
  --run-id "example_baseline_001" `
  --config "configs\default.yaml"
```

```powershell
& ".\.venv\Scripts\python.exe" src\run_video_enhanced.py `
  --video "data\raw\development\example.mp4" `
  --clip-id "example" `
  --split development `
  --run-id "example_enhanced_001" `
  --config "configs\default.yaml"
```

To make an explicit smoothing override:

```powershell
& ".\.venv\Scripts\python.exe" src\run_video_enhanced.py `
  --video "data\raw\development\example.mp4" `
  --clip-id "example" `
  --split development `
  --run-id "example_enhanced_alpha_040" `
  --config "configs\default.yaml" `
  --alpha 0.4
```

Add `--overwrite` only when intentionally replacing every output belonging to
that run ID.

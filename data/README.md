# Local dataset sources and evaluation inventory

Raw recordings under `data/raw/` are Git-ignored local inputs and are not
redistributed by this repository.

## Development dataset

The frozen development manifest is
`data/manifests/development_dataset_manifest.csv`. Its annotation CSV is
`data/annotations/development_repetition_annotations.csv`, with review and
freeze metadata in
`data/annotations/development_repetition_annotations.review.json`.

The six self-recorded filenames describe recording intent only. Recalled
attempt counts are deliberately excluded from ground truth. The historical
`setup_test_2026-07-20.mp4` remains engineering/smoke evidence and is not in the
development manifest.

## Held-out test dataset

The held-out set consists of `test01`, `test02`, `test03` and `test04` in
`data/manifests/test_dataset_manifest.csv`. Their frozen ground truth is in
`data/annotations/test_repetition_annotations.csv`, with review and freeze
metadata in `data/annotations/test_repetition_annotations.review.json`. The
collection and no-retuning rules are recorded in
`docs/held_out_test_collection_protocol.md`.

The review record has status `complete` and binds the frozen annotation
SHA-256 `ca8f93ac28095d70ce207b0bfa27b33b628f3966b90510ad78826eb32fd8e27c`.

These are fresh recordings collected after the development scientific freeze.
All four use the anonymised participant identifier `P_TEST_01`; they therefore
provide held-out evidence across clips, not participant-independent
generalisation evidence. Their raw videos remain excluded from Git under the
same `data/raw/` policy as the development recordings.

## External development source

Six selected local videos came from the Kaggle dataset **LSTM Exercise
Classification: Push Up Videos**:

- Creator: Mohamad Ashraf (Kaggle account `mohamadashrafsalama`).
- Dataset URL: https://www.kaggle.com/datasets/mohamadashrafsalama/pushup
- Licence: Creative Commons Attribution-NonCommercial-ShareAlike 4.0
  International (CC BY-NC-SA 4.0), as displayed by Kaggle:
  https://creativecommons.org/licenses/by-nc-sa/4.0/
- Local access/download date: 2026-08-11.

Suggested reference: Ashraf, M. (n.d.) *LSTM Exercise Classification: Push Up
Videos*. Kaggle. Available at:
https://www.kaggle.com/datasets/mohamadashrafsalama/pushup (Accessed: 11 August
2026).

`Correct sequence` and `Wrong sequence` are external source groupings only;
they are not this project's repetition labels and are not used to derive ground
truth. Every visible attempt is independently annotated under
`docs/manual_annotation_protocol.md`. The dataset's `labels/correct.npy` and
`labels/incorrect.npy` files are not used or tracked as project ground truth.

The manifest values `P_EXT_KAGGLE_*` are clip-local unknown-identity
surrogates. They do not establish that the six clips contain six known,
independent human participants. Participant-level independence or
generalisation must therefore not be claimed from these external clips.

Selected original source identities:

- `Correct sequence/Copy of push up 47.mp4`
- `Correct sequence/Copy of push up 80.mp4`
- `Correct sequence/Copy of push up 164.mp4`
- `Wrong sequence/8.mp4`
- `Wrong sequence/Copy of push up 42.mp4`
- `Wrong sequence/Copy of push up 81.mp4`

## Development technical inventory

Metadata was read with OpenCV without pose estimation or push-up analysis.
Duration is deterministically `frame_count / source_fps`. `decodes` records
whether OpenCV successfully decoded one source frame.

| Clip ID | SHA-256 | FPS | Frames | Resolution | Duration (s) | Decodes |
| --- | --- | ---: | ---: | --- | ---: | --- |
| `dev01_correct` | `0a346b13b7b3c68cf04bd0ab5c9cf76e0ebf86983eb369badf5380fd171ccbb8` | 30.00073621438563 | 815 | 1920x1080 | 27.166 | yes |
| `dev02_insufficient_depth` | `bd5fd047267a54e3761fffdef08720afe15d2b98c010a977d08e633bd50e3537` | 29.99873423062318 | 711 | 1920x1080 | 23.701 | yes |
| `dev03_incomplete_extension` | `2d7aee7f52b4397bedee787c7fa2f5d96f2561af06a5439793b6669f8b612b58` | 30.000712606000143 | 842 | 1920x1080 | 28.066 | yes |
| `dev04_alignment_deviation` | `56d27bcd3285120918b63d48975fbbe0991668e6f6b2365698901cc9da066153` | 30.00083104795147 | 722 | 1920x1080 | 24.066 | yes |
| `dev05_mixed_fast` | `e29278659cc5740c78e6a65efdd13535c11895c6bc9d25e18ae5a9bc306e66c2` | 30.001178828244726 | 509 | 1920x1080 | 16.966 | yes |
| `dev06_mixed_diagonal` | `419605b1ecef997792b6abef09bb9976fb089385339ee479901f7ae6c85d45c8` | 29.998617575226948 | 651 | 1920x1080 | 21.701 | yes |
| `ext_kaggle_01` | `432d676b8fcd0e5d760421a1a3c57a684e5abc1117a69db8a4929160bedbc284` | 25.0 | 95 | 640x360 | 3.800 | yes |
| `ext_kaggle_02` | `3c12c49fdbc75ac8145d121348a974cece6cf47de9794c0e43ca12e815409bfa` | 25.0 | 157 | 640x360 | 6.280 | yes |
| `ext_kaggle_03` | `0212fb967e95401e6d99471a814539d373a4b9dd4890c6936adae6753fcb6f46` | 25.0 | 115 | 640x360 | 4.600 | yes |
| `ext_kaggle_04` | `86e06da20f1cbd4bc9291e3ff3a309df5deaf87eeb2441717fe22eed975c7593` | 29.97 | 87 | 640x360 | 2.903 | yes |
| `ext_kaggle_05` | `d31caa4a2b160554e8694543e685f5733e75fb5867d781c00bd2d172933a4107` | 25.0 | 116 | 640x360 | 4.640 | yes |
| `ext_kaggle_06` | `67d5c4c422da284197ea68e0f0f00b64d1a764efaf24816b867ab80184f5fe70` | 25.0 | 93 | 640x360 | 3.720 | yes |

import json

import pytest
import yaml

import run_video
import run_video_enhanced
from config.runtime import (
    apply_cli_overrides,
    load_runtime_config,
)
from utils.paths import PROJECT_ROOT

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


def load_document():
    return yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))


def write_document(tmp_path, document):
    config_path = tmp_path / "runtime.yaml"
    config_path.write_text(
        yaml.safe_dump(document, sort_keys=False),
        encoding="utf-8",
    )
    return config_path


def test_default_config_loads_exact_current_values():
    config = load_runtime_config(DEFAULT_CONFIG_PATH)

    assert config.pose.minimum_detection_confidence == 0.5
    assert config.pose.minimum_tracking_confidence == 0.5
    assert config.features.minimum_landmark_visibility == 0.5
    assert config.features.side_acquisition_frames == 3
    assert config.features.side_switch_frames == 5
    assert config.features.side_switch_margin == 0.10
    assert config.features.missing_side_grace_frames == 5
    assert config.features.ema_alpha == 0.3
    assert config.segmentation.top_region_threshold == 130.0
    assert config.segmentation.bottom_region_threshold == 120.0
    assert config.segmentation.hysteresis == 5.0
    assert config.segmentation.phase_confirmation_frames == 3
    assert config.segmentation.missing_angle_grace_frames == 5
    assert config.segmentation.minimum_repetition_frames == 8
    assert config.classification.depth_threshold == 65.0
    assert config.classification.extension_threshold == 150.0
    assert config.classification.alignment_minimum == 160.0
    assert config.classification.alignment_deviation_min_frames == 3
    assert config.classification.alignment_deviation_min_ratio == 0.20
    assert config.classification.minimum_alignment_valid_ratio == 0.50


def test_missing_required_field_is_rejected(tmp_path):
    document = load_document()
    del document["features"]["ema_alpha"]

    with pytest.raises(
        ValueError,
        match="missing required fields.*ema_alpha",
    ):
        load_runtime_config(write_document(tmp_path, document))


@pytest.mark.parametrize(
    ("field_path", "invalid_value", "message"),
    [
        (
            ("features", "ema_alpha"),
            0.0,
            "features.ema_alpha",
        ),
        (
            ("segmentation", "phase_confirmation_frames"),
            2.5,
            "must be an integer",
        ),
        (
            (
                "classification",
                "minimum_alignment_valid_ratio",
            ),
            1.1,
            "minimum_alignment_valid_ratio",
        ),
    ],
)
def test_invalid_numeric_values_are_rejected(
    tmp_path,
    field_path,
    invalid_value,
    message,
):
    document = load_document()
    section, field = field_path
    document[section][field] = invalid_value

    with pytest.raises(ValueError, match=message):
        load_runtime_config(write_document(tmp_path, document))


def test_unknown_field_is_rejected(tmp_path):
    document = load_document()
    document["features"]["hidden_default"] = 12

    with pytest.raises(
        ValueError,
        match="unknown fields.*hidden_default",
    ):
        load_runtime_config(write_document(tmp_path, document))


def test_malformed_yaml_is_rejected(tmp_path):
    config_path = tmp_path / "malformed.yaml"
    config_path.write_text(
        "pose: [unclosed\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Malformed YAML",
    ):
        load_runtime_config(config_path)


def test_cli_override_takes_precedence_and_is_recorded():
    config = load_runtime_config(DEFAULT_CONFIG_PATH)

    resolved, overrides = apply_cli_overrides(
        config,
        ema_alpha=0.7,
    )

    assert config.features.ema_alpha == 0.3
    assert resolved.features.ema_alpha == 0.7
    assert overrides == {"features.ema_alpha": 0.7}


def test_config_serialises_to_plain_json_types():
    config = load_runtime_config(DEFAULT_CONFIG_PATH)

    encoded = json.dumps(
        config.to_dict(),
        sort_keys=True,
    )
    decoded = json.loads(encoded)

    assert decoded["config_schema_version"] == 1
    assert decoded["segmentation"]["minimum_repetition_frames"] == 8


@pytest.mark.parametrize(
    "parser",
    [
        run_video.parse_arguments,
        run_video_enhanced.parse_arguments,
    ],
)
def test_recorded_runner_rejects_unknown_split(parser):
    with pytest.raises(SystemExit):
        parser(
            [
                "--video",
                "input.mp4",
                "--clip-id",
                "clip",
                "--split",
                "calibration",
            ]
        )


def test_enhanced_alpha_defaults_to_config_not_cli():
    args = run_video_enhanced.parse_arguments(
        [
            "--video",
            "input.mp4",
            "--clip-id",
            "clip",
            "--split",
            "development",
        ]
    )

    assert args.alpha is None
    assert args.config == DEFAULT_CONFIG_PATH

"""Load strict YAML configuration into immutable runtime value objects.

This module owns schema validation, the development/test split vocabulary and
explicit CLI overrides. It requires every configured field and does not tune,
centralise or silently supply scientific defaults; the selected YAML file owns
the values used by recorded runners.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import yaml

CONFIG_SCHEMA_VERSION = 1
ALLOWED_SPLITS = ("development", "test")


@dataclass(frozen=True)
class PoseConfig:
    """Operational MediaPipe detection and tracking confidence settings."""

    minimum_detection_confidence: float
    minimum_tracking_confidence: float


@dataclass(frozen=True)
class BaselineConfig:
    """Raw baseline counting thresholds and frame-warning limits in degrees."""

    top_elbow_angle: float
    bottom_elbow_angle: float
    top_extension_warning_threshold: float
    depth_warning_threshold: float
    alignment_warning_minimum: float


@dataclass(frozen=True)
class FeatureConfig:
    """Enhanced visibility, stable-side and EMA preprocessing settings."""

    minimum_landmark_visibility: float
    side_acquisition_frames: int
    side_switch_frames: int
    side_switch_margin: float
    missing_side_grace_frames: int
    ema_alpha: float


@dataclass(frozen=True)
class SegmentationConfig:
    """Enhanced temporal phase and repetition-window settings.

    Angle values are degrees; confirmation, grace and minimum-duration values
    are frame counts.
    """

    top_region_threshold: float
    bottom_region_threshold: float
    hysteresis: float
    phase_confirmation_frames: int
    missing_angle_grace_frames: int
    minimum_repetition_frames: int


@dataclass(frozen=True)
class ClassificationConfig:
    """Enhanced repetition-rule thresholds and evidence requirements.

    Angle fields are degrees, frame fields are counts and ratio fields are
    fractions from zero to one.
    """

    depth_threshold: float
    extension_threshold: float
    alignment_minimum: float
    alignment_deviation_min_frames: int
    alignment_deviation_min_ratio: float
    minimum_alignment_valid_ratio: float


@dataclass(frozen=True)
class RuntimeConfig:
    """One fully validated immutable configuration used by a runner."""

    config_schema_version: int
    pose: PoseConfig
    baseline: BaselineConfig
    features: FeatureConfig
    segmentation: SegmentationConfig
    classification: ClassificationConfig

    def to_dict(self) -> dict[str, Any]:
        """Return all resolved sections for provenance serialization."""
        return asdict(self)


ROOT_FIELDS = frozenset(
    {
        "config_schema_version",
        "pose",
        "baseline",
        "features",
        "segmentation",
        "classification",
    }
)

SECTION_FIELDS = {
    "pose": frozenset(
        {
            "minimum_detection_confidence",
            "minimum_tracking_confidence",
        }
    ),
    "baseline": frozenset(
        {
            "top_elbow_angle",
            "bottom_elbow_angle",
            "top_extension_warning_threshold",
            "depth_warning_threshold",
            "alignment_warning_minimum",
        }
    ),
    "features": frozenset(
        {
            "minimum_landmark_visibility",
            "side_acquisition_frames",
            "side_switch_frames",
            "side_switch_margin",
            "missing_side_grace_frames",
            "ema_alpha",
        }
    ),
    "segmentation": frozenset(
        {
            "top_region_threshold",
            "bottom_region_threshold",
            "hysteresis",
            "phase_confirmation_frames",
            "missing_angle_grace_frames",
            "minimum_repetition_frames",
        }
    ),
    "classification": frozenset(
        {
            "depth_threshold",
            "extension_threshold",
            "alignment_minimum",
            "alignment_deviation_min_frames",
            "alignment_deviation_min_ratio",
            "minimum_alignment_valid_ratio",
        }
    ),
}


def _require_mapping(
    value: object,
    location: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{location} must be a YAML mapping")

    non_string_keys = [key for key in value if not isinstance(key, str)]

    if non_string_keys:
        raise ValueError(f"{location} contains non-text field names: {non_string_keys}")

    return value


def _validate_fields(
    mapping: Mapping[str, Any],
    expected: frozenset[str],
    location: str,
) -> None:
    actual = set(mapping)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)

    if missing:
        raise ValueError(f"{location} is missing required fields: {missing}")

    if unknown:
        raise ValueError(f"{location} contains unknown fields: {unknown}")


def _require_integer(
    value: object,
    location: str,
    minimum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{location} must be an integer")

    if value < minimum:
        raise ValueError(f"{location} must be at least {minimum}")

    return value


def _require_number(
    value: object,
    location: str,
    minimum: float,
    maximum: float,
    *,
    minimum_inclusive: bool = True,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location} must be a finite number")

    number = float(value)

    if not math.isfinite(number):
        raise ValueError(f"{location} must be a finite number")

    below_minimum = number < minimum if minimum_inclusive else number <= minimum

    if below_minimum or number > maximum:
        opening = "[" if minimum_inclusive else "("
        raise ValueError(f"{location} must be in {opening}{minimum}, {maximum}]")

    return number


def _require_angle(
    value: object,
    location: str,
) -> float:
    return _require_number(
        value,
        location,
        0.0,
        180.0,
    )


def _validated_section(
    root: Mapping[str, Any],
    section_name: str,
    source_name: str,
) -> Mapping[str, Any]:
    location = f"{source_name}:{section_name}"
    section = _require_mapping(
        root[section_name],
        location,
    )
    _validate_fields(
        section,
        SECTION_FIELDS[section_name],
        location,
    )
    return section


def _build_runtime_config(
    document: object,
    source_name: str,
) -> RuntimeConfig:
    """Validate one loaded document in stable schema/section/field order.

    Root shape and schema version are checked before section shapes, individual
    field ranges and cross-field threshold relationships. This ordering keeps
    invalid configuration failures deterministic and context-rich.
    """
    root = _require_mapping(document, source_name)
    _validate_fields(root, ROOT_FIELDS, source_name)

    schema_version = _require_integer(
        root["config_schema_version"],
        f"{source_name}:config_schema_version",
        minimum=1,
    )

    if schema_version != CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"{source_name}:config_schema_version must be "
            f"{CONFIG_SCHEMA_VERSION}; found {schema_version}"
        )

    pose = _validated_section(root, "pose", source_name)
    baseline = _validated_section(
        root,
        "baseline",
        source_name,
    )
    features = _validated_section(
        root,
        "features",
        source_name,
    )
    segmentation = _validated_section(
        root,
        "segmentation",
        source_name,
    )
    classification = _validated_section(
        root,
        "classification",
        source_name,
    )

    pose_config = PoseConfig(
        minimum_detection_confidence=_require_number(
            pose["minimum_detection_confidence"],
            f"{source_name}:pose.minimum_detection_confidence",
            0.0,
            1.0,
            minimum_inclusive=False,
        ),
        minimum_tracking_confidence=_require_number(
            pose["minimum_tracking_confidence"],
            f"{source_name}:pose.minimum_tracking_confidence",
            0.0,
            1.0,
            minimum_inclusive=False,
        ),
    )

    baseline_config = BaselineConfig(
        top_elbow_angle=_require_angle(
            baseline["top_elbow_angle"],
            f"{source_name}:baseline.top_elbow_angle",
        ),
        bottom_elbow_angle=_require_angle(
            baseline["bottom_elbow_angle"],
            f"{source_name}:baseline.bottom_elbow_angle",
        ),
        top_extension_warning_threshold=_require_angle(
            baseline["top_extension_warning_threshold"],
            (f"{source_name}:baseline.top_extension_warning_threshold"),
        ),
        depth_warning_threshold=_require_angle(
            baseline["depth_warning_threshold"],
            f"{source_name}:baseline.depth_warning_threshold",
        ),
        alignment_warning_minimum=_require_angle(
            baseline["alignment_warning_minimum"],
            f"{source_name}:baseline.alignment_warning_minimum",
        ),
    )

    if baseline_config.bottom_elbow_angle >= baseline_config.top_elbow_angle:
        raise ValueError(
            f"{source_name}:baseline.bottom_elbow_angle must be "
            "lower than baseline.top_elbow_angle"
        )

    feature_config = FeatureConfig(
        minimum_landmark_visibility=_require_number(
            features["minimum_landmark_visibility"],
            (f"{source_name}:features.minimum_landmark_visibility"),
            0.0,
            1.0,
        ),
        side_acquisition_frames=_require_integer(
            features["side_acquisition_frames"],
            f"{source_name}:features.side_acquisition_frames",
            minimum=1,
        ),
        side_switch_frames=_require_integer(
            features["side_switch_frames"],
            f"{source_name}:features.side_switch_frames",
            minimum=1,
        ),
        side_switch_margin=_require_number(
            features["side_switch_margin"],
            f"{source_name}:features.side_switch_margin",
            0.0,
            1.0,
        ),
        missing_side_grace_frames=_require_integer(
            features["missing_side_grace_frames"],
            (f"{source_name}:features.missing_side_grace_frames"),
            minimum=0,
        ),
        ema_alpha=_require_number(
            features["ema_alpha"],
            f"{source_name}:features.ema_alpha",
            0.0,
            1.0,
            minimum_inclusive=False,
        ),
    )

    segmentation_config = SegmentationConfig(
        top_region_threshold=_require_angle(
            segmentation["top_region_threshold"],
            (f"{source_name}:segmentation.top_region_threshold"),
        ),
        bottom_region_threshold=_require_angle(
            segmentation["bottom_region_threshold"],
            (f"{source_name}:segmentation.bottom_region_threshold"),
        ),
        hysteresis=_require_angle(
            segmentation["hysteresis"],
            f"{source_name}:segmentation.hysteresis",
        ),
        phase_confirmation_frames=_require_integer(
            segmentation["phase_confirmation_frames"],
            (f"{source_name}:segmentation.phase_confirmation_frames"),
            minimum=1,
        ),
        missing_angle_grace_frames=_require_integer(
            segmentation["missing_angle_grace_frames"],
            (f"{source_name}:segmentation.missing_angle_grace_frames"),
            minimum=0,
        ),
        minimum_repetition_frames=_require_integer(
            segmentation["minimum_repetition_frames"],
            (f"{source_name}:segmentation.minimum_repetition_frames"),
            minimum=1,
        ),
    )

    if (
        segmentation_config.bottom_region_threshold
        >= segmentation_config.top_region_threshold
    ):
        raise ValueError(
            f"{source_name}:segmentation.bottom_region_threshold "
            "must be lower than "
            "segmentation.top_region_threshold"
        )

    classification_config = ClassificationConfig(
        depth_threshold=_require_angle(
            classification["depth_threshold"],
            f"{source_name}:classification.depth_threshold",
        ),
        extension_threshold=_require_angle(
            classification["extension_threshold"],
            f"{source_name}:classification.extension_threshold",
        ),
        alignment_minimum=_require_angle(
            classification["alignment_minimum"],
            f"{source_name}:classification.alignment_minimum",
        ),
        alignment_deviation_min_frames=_require_integer(
            classification["alignment_deviation_min_frames"],
            (f"{source_name}:classification.alignment_deviation_min_frames"),
            minimum=1,
        ),
        alignment_deviation_min_ratio=_require_number(
            classification["alignment_deviation_min_ratio"],
            (f"{source_name}:classification.alignment_deviation_min_ratio"),
            0.0,
            1.0,
        ),
        minimum_alignment_valid_ratio=_require_number(
            classification["minimum_alignment_valid_ratio"],
            (f"{source_name}:classification.minimum_alignment_valid_ratio"),
            0.0,
            1.0,
        ),
    )

    return RuntimeConfig(
        config_schema_version=schema_version,
        pose=pose_config,
        baseline=baseline_config,
        features=feature_config,
        segmentation=segmentation_config,
        classification=classification_config,
    )


def load_runtime_config(
    config_path: str | Path,
) -> RuntimeConfig:
    """Load UTF-8 YAML and return its strictly validated runtime configuration.

    Missing files raise ``FileNotFoundError`` and malformed YAML or schema/value
    failures raise ``ValueError``. Unknown and missing fields are rejected rather
    than ignored or filled with implicit defaults.
    """
    path = Path(config_path)

    try:
        with path.open(encoding="utf-8") as config_file:
            document = yaml.safe_load(config_file)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Runtime configuration does not exist: {path}"
        ) from error
    except yaml.YAMLError as error:
        raise ValueError(
            f"Malformed YAML in runtime configuration {path}: {error}"
        ) from error

    return _build_runtime_config(
        document,
        source_name=str(path),
    )


def apply_cli_overrides(
    config: RuntimeConfig,
    *,
    ema_alpha: float | None = None,
) -> tuple[RuntimeConfig, dict[str, float]]:
    """Return a copied configuration plus explicit provenance overrides.

    Currently only the enhanced EMA alpha may be overridden. ``None`` returns
    the original immutable configuration and an empty override mapping.
    """
    if ema_alpha is None:
        return config, {}

    validated_alpha = _require_number(
        ema_alpha,
        "--alpha",
        0.0,
        1.0,
        minimum_inclusive=False,
    )
    overridden_features = replace(
        config.features,
        ema_alpha=validated_alpha,
    )
    return (
        replace(config, features=overridden_features),
        {"features.ema_alpha": validated_alpha},
    )

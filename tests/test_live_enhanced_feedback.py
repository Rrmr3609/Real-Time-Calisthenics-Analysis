from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

import run_live_enhanced


def make_classification(
    predicted_class,
    *,
    rep_id=1,
    triggered_rules=(),
    alignment_valid_ratio=1.0,
):
    return SimpleNamespace(
        rep_id=rep_id,
        predicted_class=predicted_class,
        triggered_rules=tuple(triggered_rules),
        alignment_valid_ratio=alignment_valid_ratio,
    )


@pytest.mark.parametrize(
    ("predicted_class", "expected_category", "expected_message"),
    [
        ("correct", "FORM CRITERIA MET", "Rep meets the project form criteria."),
        (
            "insufficient_depth",
            "DEPTH",
            "Try lowering further before returning to the top.",
        ),
        (
            "incomplete_extension",
            "EXTENSION",
            "Extend the elbows more fully at the top.",
        ),
        (
            "alignment_deviation",
            "BODY ALIGNMENT",
            "Keep the shoulders, hips and ankles more aligned.",
        ),
        (
            "unscorable",
            "NOT SCORED",
            "Improve body visibility and remain roughly side-on.",
        ),
    ],
)
def test_completed_classes_have_readable_primary_feedback(
    predicted_class,
    expected_category,
    expected_message,
):
    content = run_live_enhanced.build_completed_feedback(
        make_classification(predicted_class),
        minimum_alignment_valid_ratio=0.5,
    )

    assert content.headline == f"REP 1 — {expected_category}"
    assert content.primary == expected_message
    primary_text = f"{content.headline} {content.primary}"
    assert predicted_class not in primary_text.lower()
    assert "_" not in primary_text


def test_unscorable_feedback_requests_better_visibility():
    content = run_live_enhanced.build_completed_feedback(
        make_classification(
            "unscorable",
            alignment_valid_ratio=0.2,
        ),
        minimum_alignment_valid_ratio=0.5,
    )

    assert "NOT SCORED" in content.headline
    assert "visibility" in content.primary
    assert content.secondary == (
        "Alignment not assessed: insufficient body visibility",
    )
    assert content.tone == "neutral"


def test_additional_triggered_rules_are_reported_without_identifiers():
    content = run_live_enhanced.build_completed_feedback(
        make_classification(
            "insufficient_depth",
            triggered_rules=(
                "insufficient_depth",
                "incomplete_extension",
                "alignment_deviation",
            ),
        ),
        minimum_alignment_valid_ratio=0.5,
    )

    assert content.secondary == (
        "Also observed: incomplete elbow extension, body alignment deviation",
    )
    assert "_" not in content.secondary[0]


def test_unavailable_alignment_is_not_described_as_good():
    content = run_live_enhanced.build_completed_feedback(
        make_classification(
            "insufficient_depth",
            triggered_rules=("insufficient_depth",),
            alignment_valid_ratio=0.2,
        ),
        minimum_alignment_valid_ratio=0.5,
    )

    assert content.secondary == (
        "Alignment not assessed: insufficient body visibility",
    )
    assert "good" not in " ".join(content.secondary).lower()


def test_latest_feedback_persists_then_expires():
    presenter = run_live_enhanced.TimedFeedbackPresenter(duration_seconds=2.0)
    first = run_live_enhanced.build_completed_feedback(
        make_classification("correct"),
        minimum_alignment_valid_ratio=0.5,
    )
    second = run_live_enhanced.build_completed_feedback(
        make_classification("incomplete_extension", rep_id=2),
        minimum_alignment_valid_ratio=0.5,
    )

    presenter.record(
        first,
        now=10.0,
    )
    assert presenter.current(11.99).primary == ("Rep meets the project form criteria.")

    presenter.record(
        second,
        now=11.0,
    )
    assert presenter.current(12.99).primary == (
        "Extend the elbows more fully at the top."
    )
    assert presenter.current(12.99).headline == "REP 2 — EXTENSION"
    assert presenter.current(13.0) is None


@pytest.mark.parametrize(
    ("phase", "side", "elbow_valid", "expected"),
    [
        (
            "top",
            "none",
            False,
            "Move fully into view and remain roughly side-on",
        ),
        ("top", "left", True, "Ready — begin a push-up"),
        ("descending", "left", True, "Tracking repetition..."),
        (
            "ascending",
            "right",
            True,
            "Return to the top to complete the repetition",
        ),
    ],
)
def test_neutral_guidance_is_tracking_not_form_judgement(
    phase,
    side,
    elbow_valid,
    expected,
):
    assert (
        run_live_enhanced.neutral_guidance(
            phase=phase,
            selected_side=side,
            elbow_feature_valid=elbow_valid,
        )
        == expected
    )


def make_session(*classifications):
    session = run_live_enhanced.LiveSessionResults(
        started_at_utc=datetime(2026, 8, 11, 12, 30, tzinfo=timezone.utc)
    )
    for classification in classifications:
        session.record(
            classification,
            minimum_alignment_valid_ratio=0.5,
        )
    return session


def test_session_class_counts_include_all_supported_outcomes():
    session = make_session(
        make_classification("correct", rep_id=1),
        make_classification("correct", rep_id=2),
        make_classification("insufficient_depth", rep_id=3),
        make_classification("incomplete_extension", rep_id=4),
        make_classification("alignment_deviation", rep_id=5),
        make_classification(
            "unscorable",
            rep_id=6,
            alignment_valid_ratio=0.1,
        ),
    )

    assert session.total_repetitions == 6
    assert session.class_counts() == {
        "correct": 2,
        "insufficient_depth": 1,
        "incomplete_extension": 1,
        "alignment_deviation": 1,
        "unscorable": 1,
    }


def test_zero_repetition_summary_is_sensible():
    lines = run_live_enhanced.build_session_summary_lines(make_session())

    assert "Completed repetitions: 0" in lines
    assert "Repetitions meeting project criteria: 0" in lines
    assert "No completed repetitions were detected." in lines
    assert not any("0/0" in line for line in lines)


def test_per_repetition_summary_uses_readable_categories():
    session = make_session(
        make_classification("correct", rep_id=1),
        make_classification("insufficient_depth", rep_id=2),
    )

    lines = run_live_enhanced.build_session_summary_lines(session)

    assert "Rep 1: FORM CRITERIA MET" in lines
    assert "Rep 2: DEPTH" in lines
    heading_index = lines.index("PER-REPETITION SUMMARY")
    assert lines[heading_index - 1] == ""
    assert not any("insufficient_depth" in line for line in lines)


def test_report_contains_human_feedback_and_evidence_status():
    session = make_session(
        make_classification(
            "insufficient_depth",
            rep_id=3,
            triggered_rules=("insufficient_depth", "incomplete_extension"),
            alignment_valid_ratio=0.2,
        )
    )

    report = run_live_enhanced.render_live_session_report(
        session,
        config_identity="configs/default.yaml",
        config_sha256="a" * 64,
    )

    assert "not formal evaluation evidence" in report
    assert "Configuration: configs/default.yaml" in report
    assert f"Configuration SHA-256: {'a' * 64}" in report
    assert "Rep 3 — Insufficient depth" in report
    assert "Try lowering further before returning to the top." in report
    assert "Additional observations: incomplete elbow extension" in report
    assert "Alignment evidence: Not assessed" in report
    assert "insufficient_depth" not in report


def test_report_omits_redundant_optional_fields_and_separates_repetitions():
    session = make_session(
        make_classification("correct", rep_id=1),
        make_classification("incomplete_extension", rep_id=2),
    )

    report = run_live_enhanced.render_live_session_report(
        session,
        config_identity="configs/default.yaml",
        config_sha256="c" * 64,
    )

    assert "Rep 1 — Meets project criteria" in report
    assert "Rep 2 — Incomplete extension" in report
    assert "Additional observations: None" not in report
    assert "Alignment evidence: Available" not in report
    assert (
        "Feedback: Rep meets the project form criteria.\n\nRep 2 — Incomplete extension"
    ) in report


def test_report_path_is_collision_safe(tmp_path):
    session = make_session()
    kwargs = {
        "config_identity": "configs/default.yaml",
        "config_sha256": "b" * 64,
        "output_dir": tmp_path,
    }

    first_path = run_live_enhanced.save_live_session_report(session, **kwargs)
    second_path = run_live_enhanced.save_live_session_report(session, **kwargs)

    assert first_path != second_path
    assert first_path.name == "enhanced_live_20260811T123000000000Z.txt"
    assert second_path.name == "enhanced_live_20260811T123000000000Z_02.txt"
    assert first_path.read_text(encoding="utf-8") == second_path.read_text(
        encoding="utf-8"
    )


@pytest.mark.parametrize(
    ("frame_width", "frame_height", "expected"),
    [
        (640, 480, (640, 480)),
        (1280, 720, (1280, 720)),
        (3840, 2160, (1280, 720)),
        (1920, 1200, (1280, 800)),
    ],
)
def test_presentation_window_size_preserves_aspect_without_upscaling(
    frame_width,
    frame_height,
    expected,
):
    assert (
        run_live_enhanced.presentation_window_size(frame_width, frame_height)
        == expected
    )


def test_zero_and_single_page_summary_bounds_are_simple():
    empty_page = run_live_enhanced.build_summary_page(
        make_session(),
        viewport_height=720,
        requested_index=0,
    )
    single_page = run_live_enhanced.build_summary_page(
        make_session(make_classification("correct", rep_id=1)),
        viewport_height=720,
        requested_index=5,
    )

    assert empty_page.page_count == 1
    assert empty_page.first_position == 0
    assert empty_page.last_position == 0
    assert empty_page.repetitions == ()
    assert single_page.page_count == 1
    assert single_page.index == 0
    assert [item.rep_id for item in single_page.repetitions] == [1]


def test_long_summary_is_paginated_to_viewport_capacity():
    session = make_session(
        *(make_classification("correct", rep_id=index) for index in range(1, 25))
    )

    first_page = run_live_enhanced.build_summary_page(
        session,
        viewport_height=720,
        requested_index=0,
    )
    last_page = run_live_enhanced.build_summary_page(
        session,
        viewport_height=720,
        requested_index=99,
    )

    assert first_page.rows_per_page == 10
    assert first_page.page_count == 3
    assert (first_page.first_position, first_page.last_position) == (1, 10)
    assert [item.rep_id for item in first_page.repetitions] == list(range(1, 11))
    assert last_page.index == 2
    assert (last_page.first_position, last_page.last_position) == (21, 24)
    assert [item.rep_id for item in last_page.repetitions] == list(range(21, 25))


@pytest.mark.parametrize("key_code", [0x280000, 0x220000, 65364, 65366])
def test_summary_next_page_navigation_is_bounded(key_code):
    assert (
        run_live_enhanced.navigate_summary_page(
            0,
            page_count=3,
            key_code=key_code,
        )
        == 1
    )
    assert (
        run_live_enhanced.navigate_summary_page(
            2,
            page_count=3,
            key_code=key_code,
        )
        == 2
    )


@pytest.mark.parametrize("key_code", [0x260000, 0x210000, 65362, 65365])
def test_summary_previous_page_navigation_is_bounded(key_code):
    assert (
        run_live_enhanced.navigate_summary_page(
            2,
            page_count=3,
            key_code=key_code,
        )
        == 1
    )
    assert (
        run_live_enhanced.navigate_summary_page(
            0,
            page_count=3,
            key_code=key_code,
        )
        == 0
    )


@pytest.mark.parametrize(
    "key_code",
    [ord("q"), ord("Q"), 27, 0x100000 | ord("q")],
)
def test_live_exit_keys_are_recognised(key_code):
    assert run_live_enhanced.exit_key_requested(key_code) is True


@pytest.mark.parametrize("key_code", [-1, ord("x"), 10, 13])
def test_non_exit_keys_do_not_finish_live_session(key_code):
    assert run_live_enhanced.exit_key_requested(key_code) is False


@pytest.mark.parametrize("key_code", [ord("q"), ord("Q"), 27, 10, 13])
def test_summary_exit_keys_are_recognised(key_code):
    assert run_live_enhanced.exit_key_requested(key_code, allow_enter=True) is True


def test_tracking_status_does_not_expose_selected_side():
    assert (
        run_live_enhanced.tracking_status(
            selected_side="none",
            elbow_feature_valid=False,
        )
        == "Tracking: Finding body position"
    )
    status = run_live_enhanced.tracking_status(
        selected_side="left",
        elbow_feature_valid=True,
    )
    assert status == "Tracking: Body position found"
    assert "left" not in status.lower()


def test_live_main_releases_resources_on_capture_failure(monkeypatch):
    camera = SimpleNamespace(
        opened=False,
        released=False,
    )

    def open_camera():
        camera.opened = True

    def release_camera():
        camera.released = True

    camera.open = open_camera
    camera.release = release_camera
    camera.read = lambda: None

    pose_estimator = SimpleNamespace(closed=False)
    pose_estimator.close = lambda: setattr(pose_estimator, "closed", True)
    destroyed = []

    monkeypatch.setattr(
        run_live_enhanced,
        "parse_arguments",
        lambda: SimpleNamespace(
            camera_index=0,
            config=run_live_enhanced.DEFAULT_CONFIG_PATH,
            alpha=None,
        ),
    )
    monkeypatch.setattr(
        run_live_enhanced,
        "WebcamCapture",
        lambda **_kwargs: camera,
    )
    monkeypatch.setattr(
        run_live_enhanced,
        "PoseEstimator",
        lambda **_kwargs: pose_estimator,
    )
    monkeypatch.setattr(
        run_live_enhanced.cv2,
        "destroyAllWindows",
        lambda: destroyed.append(True),
    )
    monkeypatch.setattr(run_live_enhanced.cv2, "namedWindow", lambda *_args: None)

    with pytest.raises(RuntimeError, match="Failed to read a frame"):
        run_live_enhanced.main()

    assert camera.opened is True
    assert camera.released is True
    assert pose_estimator.closed is True
    assert destroyed == [True]


def test_live_main_handles_unavailable_camera_without_pose_setup(
    monkeypatch,
    capsys,
):
    camera = SimpleNamespace(released=False)
    camera.open = lambda: (_ for _ in ()).throw(
        RuntimeError("Could not open camera with device index 4")
    )
    camera.release = lambda: setattr(camera, "released", True)
    destroyed = []

    monkeypatch.setattr(
        run_live_enhanced,
        "parse_arguments",
        lambda: SimpleNamespace(
            camera_index=4,
            config=run_live_enhanced.DEFAULT_CONFIG_PATH,
            alpha=None,
        ),
    )
    monkeypatch.setattr(run_live_enhanced, "WebcamCapture", lambda **_kwargs: camera)
    monkeypatch.setattr(
        run_live_enhanced,
        "PoseEstimator",
        lambda **_kwargs: pytest.fail("Pose setup must not run without a camera"),
    )
    monkeypatch.setattr(
        run_live_enhanced.cv2,
        "destroyAllWindows",
        lambda: destroyed.append(True),
    )

    assert run_live_enhanced.main() == 1

    assert capsys.readouterr().err.strip() == (
        "Unable to open camera 4. Check that a webcam is connected and not being "
        "used by another application."
    )
    assert camera.released is True
    assert destroyed == [True]


def test_live_main_does_not_swallow_unexpected_camera_error(monkeypatch):
    camera = SimpleNamespace(released=False)
    camera.open = lambda: (_ for _ in ()).throw(RuntimeError("driver bug"))
    camera.release = lambda: setattr(camera, "released", True)

    monkeypatch.setattr(
        run_live_enhanced,
        "parse_arguments",
        lambda: SimpleNamespace(
            camera_index=0,
            config=run_live_enhanced.DEFAULT_CONFIG_PATH,
            alpha=None,
        ),
    )
    monkeypatch.setattr(run_live_enhanced, "WebcamCapture", lambda **_kwargs: camera)
    monkeypatch.setattr(run_live_enhanced.cv2, "destroyAllWindows", lambda: None)

    with pytest.raises(RuntimeError, match="driver bug"):
        run_live_enhanced.main()

    assert camera.released is True

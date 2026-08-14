"""Finalise return-top extension without changing repetition detection."""

from dataclasses import replace
from typing import Optional

from analysis.repetition_result import CompletedRepetition


class ReturnTopPeakFinalizer:
    """Hold one detected repetition until its returned top phase ends.

    The phase state machine fixes the detection event and ``end_frame`` when
    return-to-top confirmation succeeds. Classification is delayed separately
    while that stable top phase remains active, allowing later valid elbow
    observations to improve the representative return-extension peak. A
    confirmed transition away from top, or an explicit stream-end flush,
    finalises the pending repetition without changing its detection fields.
    """

    def __init__(self) -> None:
        self._pending: Optional[CompletedRepetition] = None
        self._return_top_peak: Optional[float] = None

    def update(
        self,
        *,
        detected_repetition: Optional[CompletedRepetition],
        elbow_angle: Optional[float],
        returned_top_phase_active: bool,
    ) -> Optional[CompletedRepetition]:
        """Observe one frame and emit a repetition once returned top ends."""
        if detected_repetition is not None:
            if self._pending is not None:
                raise RuntimeError(
                    "A new repetition completed before the prior return-top "
                    "measurement was finalised"
                )
            self._pending = detected_repetition
            self._return_top_peak = float(detected_repetition.end_top_angle)

        if self._pending is None:
            return None

        if returned_top_phase_active:
            if elbow_angle is not None:
                observed_angle = float(elbow_angle)
                self._return_top_peak = max(
                    self._return_top_peak,
                    observed_angle,
                )
            return None

        return self._take_pending()

    def flush(self) -> Optional[CompletedRepetition]:
        """Finalise the last pending repetition when a stream ends."""
        if self._pending is None:
            return None
        return self._take_pending()

    def _take_pending(self) -> CompletedRepetition:
        if self._pending is None or self._return_top_peak is None:
            raise RuntimeError("No pending return-top measurement to finalise")

        finalised = replace(
            self._pending,
            end_top_angle=self._return_top_peak,
        )
        self._pending = None
        self._return_top_peak = None
        return finalised

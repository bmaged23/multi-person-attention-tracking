"""Original project's weighted attention formula and label thresholds."""

import numpy as np


class AttentionState:
    """Keep the tracking API while applying the original per-observation rule."""

    def unknown(self):
        return {"label": "no_face", "is_attentive": None, "score": 0.0, "signals": None}

    def update(self, signals, now, settings):
        if signals is None:
            return self.unknown()
        pitch, yaw = signals.get("pitch"), signals.get("yaw")
        head_score = 0.5
        if pitch is not None and yaw is not None:
            yaw_ok = abs(yaw) <= settings.yaw_limit
            pitch_ok = settings.pitch_down <= pitch <= settings.pitch_up
            head_score = (
                1.0 if yaw_ok and pitch_ok else 0.5 if yaw_ok or pitch_ok else 0.0
            )
        ear_value = signals.get("ear")
        ear_score = (
            0.5
            if ear_value is None
            else 0.0
            if ear_value < settings.ear_closed
            else 1.0
        )
        drowsy = ear_value is not None and ear_value < settings.ear_closed
        iris_value = signals.get("iris_avg")
        iris_score = (
            0.5
            if iris_value is None
            else 1.0
            if settings.iris_side <= iris_value <= 1 - settings.iris_side
            else 0.3
        )
        score = round(
            float(
                np.clip(
                    settings.head_weight * head_score
                    + settings.eye_weight * ear_score
                    + settings.gaze_weight * iris_score
                    + settings.visibility_weight,
                    0,
                    1,
                )
            ),
            3,
        )
        label = (
            "drowsy"
            if drowsy
            else "attentive"
            if score >= settings.attentive_min
            else "partial"
            if score >= settings.partial_min
            else "non_attentive"
        )
        return {
            "label": label,
            "is_attentive": label == "attentive",
            "score": score,
            "signals": {
                k: round(v, 4) if v is not None else None for k, v in signals.items()
            },
        }

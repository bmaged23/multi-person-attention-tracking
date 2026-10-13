"""One-to-one spatial tracks with session-local IDs and optional display names."""

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from attention_tracking.analysis.scoring import AttentionState
from attention_tracking.geometry import bbox_iou


@dataclass
class Track:
    id: str
    name: str
    bbox: tuple
    last_seen: float
    attention: AttentionState = field(default_factory=AttentionState)


class Tracker:
    def __init__(self, settings, names=None):
        self.settings, self.names = settings, names or {}
        self.tracks = {}
        self.next_id = 1

    def update(self, faces, now, shape):
        self.tracks = {
            key: t
            for key, t in self.tracks.items()
            if now - t.last_seen <= self.settings.track_timeout
        }
        current = list(self.tracks.values())
        diagonal = float(np.hypot(shape[0], shape[1]))
        costs = np.full((len(current), len(faces)), 1e6)
        for i, track in enumerate(current):
            for j, face in enumerate(faces):
                distance = (
                    np.linalg.norm(
                        (np.array(track.bbox[:2]) + track.bbox[2:]) / 2
                        - (np.array(face.bbox[:2]) + face.bbox[2:]) / 2
                    )
                    / diagonal
                )
                overlap = bbox_iou(track.bbox, face.bbox)
                if overlap >= self.settings.iou_threshold:
                    costs[i, j] = (1 - overlap) + distance
        assigned = {}
        if costs.size:
            for i, j in zip(*linear_sum_assignment(costs)):
                if costs[i, j] < 1e6:
                    assigned[j] = current[i]
        observed = []
        for j, face in enumerate(faces):
            track = assigned.get(j)
            if track is None:
                key = f"person_{self.next_id:03d}"
                track = Track(
                    key, self.names.get(key, f"Person {self.next_id}"), face.bbox, now
                )
                self.next_id += 1
                self.tracks[key] = track
            track.bbox, track.last_seen = face.bbox, now
            observed.append((track, face))
        visible = {t.id for t, _ in observed}
        for track in self.tracks.values():
            if track.id not in visible:
                track.attention.unknown()
        return observed

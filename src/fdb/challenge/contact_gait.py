from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ContactStep:
    side: str
    lift_frame: int
    touchdown_frame: int
    air_time_s: float
    swing_clearance_m: float
    forward_placement_m: float


@dataclass(frozen=True)
class ContactGaitResult:
    duration_s: float
    alternating_steps: int
    completed_strides: int
    left_steps: int
    right_steps: int
    minimum_air_time_s: float
    minimum_swing_clearance_m: float
    minimum_forward_placement_m: float
    pelvis_path_length_m: float
    net_pelvis_displacement_m: float
    success: bool


def _debounce_contacts(contacts: np.ndarray, frames: int) -> np.ndarray:
    """Remove short contact spikes without moving a sustained transition."""
    raw = np.asarray(contacts, dtype=bool)
    if frames <= 1:
        return raw.copy()
    result = raw.copy()
    for side in range(raw.shape[1]):
        start = 0
        while start < len(raw):
            end = start + 1
            while end < len(raw) and raw[end, side] == raw[start, side]:
                end += 1
            if end - start < frames:
                before = result[start - 1, side] if start else None
                after = raw[end, side] if end < len(raw) else None
                if before is not None and before == after:
                    result[start:end, side] = before
            start = end
    return result


def detect_contact_steps(
    foot_positions: np.ndarray,
    contacts: np.ndarray,
    root_xy: np.ndarray,
    frequency: float,
    minimum_air_time_s: float = 0.15,
    debounce_time_s: float = 0.025,
) -> list[ContactStep]:
    """Detect physical lift-off and touchdown events from floor contact.

    Height is measured relative to each foot's median contacted height, so the
    criterion works with different sole/body origins. Direction is the net
    pelvis travel direction, which prevents a stationary leg swing from being
    mistaken for walking.
    """
    feet = np.asarray(foot_positions, dtype=float)
    root = np.asarray(root_xy, dtype=float)
    contact = np.asarray(contacts, dtype=bool)
    if feet.ndim != 3 or feet.shape[1:] != (2, 3):
        raise ValueError("foot_positions must have shape [frames, 2, 3]")
    if contact.shape != feet.shape[:2] or root.shape != (len(feet), 2):
        raise ValueError("contacts/root_xy length or shape mismatch")
    if frequency <= 0:
        raise ValueError("frequency must be positive")

    contact = _debounce_contacts(contact, max(1, round(debounce_time_s * frequency)))
    min_air_frames = max(1, round(minimum_air_time_s * frequency))
    net = root[-1] - root[0]
    if np.linalg.norm(net) < 1e-6:
        net = np.array([1.0, 0.0])
    direction = net / np.linalg.norm(net)
    events: list[ContactStep] = []

    for side_index, side in enumerate(("left", "right")):
        foot = feet[:, side_index]
        grounded_z = foot[contact[:, side_index], 2]
        baseline_z = float(np.median(grounded_z)) if len(grounded_z) else float(np.min(foot[:, 2]))
        lift: int | None = None
        for frame in range(1, len(feet)):
            was_grounded = bool(contact[frame - 1, side_index])
            is_grounded = bool(contact[frame, side_index])
            if was_grounded and not is_grounded:
                lift = frame
            elif not was_grounded and is_grounded and lift is not None:
                if frame - lift >= min_air_frames:
                    segment = foot[lift:frame + 1]
                    events.append(ContactStep(
                        side=side,
                        lift_frame=lift,
                        touchdown_frame=frame,
                        air_time_s=(frame - lift) / frequency,
                        swing_clearance_m=float(np.max(segment[:, 2]) - baseline_z),
                        forward_placement_m=float((foot[frame, :2] - foot[lift, :2]) @ direction),
                    ))
                lift = None
    return sorted(events, key=lambda event: event.touchdown_frame)


def longest_valid_alternating_run(
    steps: list[ContactStep],
    frequency: float,
    minimum_clearance_m: float = 0.010,
    minimum_forward_placement_m: float = 0.080,
    maximum_step_gap_s: float = 1.6,
) -> list[ContactStep]:
    """Return the longest alternating sequence whose every step is meaningful."""
    valid = [
        step for step in steps
        if step.swing_clearance_m >= minimum_clearance_m
        and step.forward_placement_m >= minimum_forward_placement_m
    ]
    maximum_gap = round(maximum_step_gap_s * frequency)
    best: list[ContactStep] = []
    current: list[ContactStep] = []
    for step in valid:
        if not current or (
            step.side != current[-1].side
            and 0 < step.touchdown_frame - current[-1].touchdown_frame <= maximum_gap
        ):
            current.append(step)
        else:
            current = [step]
        if len(current) > len(best):
            best = current.copy()
    return best


def evaluate_contact_gait(
    foot_positions: np.ndarray,
    contacts: np.ndarray,
    root_xy: np.ndarray,
    frequency: float,
    required_steps: int = 10,
    minimum_duration_s: float = 5.0,
    minimum_path_m: float = 1.0,
) -> tuple[ContactGaitResult, list[ContactStep], list[ContactStep]]:
    steps = detect_contact_steps(foot_positions, contacts, root_xy, frequency)
    alternating = longest_valid_alternating_run(steps, frequency)
    root = np.asarray(root_xy, dtype=float)
    path = float(np.sum(np.linalg.norm(np.diff(root, axis=0), axis=1))) if len(root) > 1 else 0.0
    displacement = float(np.linalg.norm(root[-1] - root[0])) if len(root) else 0.0
    duration = max(0, len(root) - 1) / frequency
    selected = alternating[:required_steps]
    success = bool(
        len(alternating) >= required_steps
        and duration >= minimum_duration_s
        and path >= minimum_path_m
        and {step.side for step in selected} == {"left", "right"}
    )
    values = selected or alternating or steps
    result = ContactGaitResult(
        duration_s=duration,
        alternating_steps=len(alternating),
        completed_strides=len(alternating) // 2,
        left_steps=sum(step.side == "left" for step in alternating),
        right_steps=sum(step.side == "right" for step in alternating),
        minimum_air_time_s=min((step.air_time_s for step in values), default=0.0),
        minimum_swing_clearance_m=min((step.swing_clearance_m for step in values), default=0.0),
        minimum_forward_placement_m=min((step.forward_placement_m for step in values), default=0.0),
        pelvis_path_length_m=path,
        net_pelvis_displacement_m=displacement,
        success=success,
    )
    return result, steps, alternating

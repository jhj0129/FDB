import numpy as np

from fdb.challenge.contact_gait import detect_contact_steps, evaluate_contact_gait


def _synthetic_walk(step_count: int = 10, frequency: float = 100.0):
    frames = 650
    feet = np.zeros((frames, 2, 3), dtype=float)
    contacts = np.ones((frames, 2), dtype=bool)
    root = np.zeros((frames, 2), dtype=float)
    root[:, 0] = np.linspace(0.0, 2.5, frames)
    feet[:, 1, 0] = 0.2
    for index in range(step_count):
        side = index % 2
        lift = 30 + 55 * index
        land = lift + 32
        contacts[lift:land, side] = False
        phase = np.linspace(0.0, np.pi, land - lift)
        feet[lift:land, side, 2] = 0.04 * np.sin(phase)
        start_x = float(feet[lift - 1, side, 0])
        end_x = start_x + 0.22
        feet[lift:land, side, 0] = np.linspace(start_x, end_x, land - lift)
        feet[land:, side, 0] = end_x
    return feet, contacts, root, frequency


def test_contact_gait_accepts_ten_physical_alternating_steps():
    feet, contacts, root, frequency = _synthetic_walk()
    result, steps, alternating = evaluate_contact_gait(feet, contacts, root, frequency)
    assert len(steps) == 10
    assert len(alternating) == 10
    assert result.completed_strides == 5
    assert result.success


def test_contact_gait_rejects_short_contact_jitter():
    feet, contacts, root, frequency = _synthetic_walk(step_count=0)
    contacts[100:102, 0] = False
    feet[100:102, 0, 2] = 0.003
    assert detect_contact_steps(feet, contacts, root, frequency) == []


def test_contact_gait_rejects_in_place_leg_motion():
    feet, contacts, root, frequency = _synthetic_walk()
    feet[:, :, 0] = 0.0
    result, _, alternating = evaluate_contact_gait(feet, contacts, root, frequency)
    assert alternating == []
    assert not result.success

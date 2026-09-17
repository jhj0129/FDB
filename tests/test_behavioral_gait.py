import numpy as np

from fdb.challenge.behavioral_gait import evaluate_behavioral_walk


def _synthetic_walk(step_count: int = 12, frequency: int = 50):
    frames = step_count * 25 + 40
    sites = np.zeros((frames, 7, 3), dtype=float)
    sites[:, 3, 2] = 0.045
    sites[:, 6, 2] = 0.045
    root = np.column_stack((np.arange(frames) * 0.012, np.zeros(frames)))
    for step in range(step_count):
        side = 6 if step % 2 == 0 else 3
        start = 10 + step * 25
        stop = start + 18
        phase = np.linspace(0.0, np.pi, stop - start)
        sites[start:stop, side, 2] = 0.045 + 0.10 * np.sin(phase)
        sites[start:stop, side, 0] = root[start:stop, 0] + np.linspace(0.0, 0.24, stop - start)
    return sites, root, frequency


def test_real_walking_requires_ten_alternating_forward_touchdowns():
    sites, root, frequency = _synthetic_walk()
    result, events = evaluate_behavioral_walk(sites, root, frequency)
    assert result.success
    assert result.alternating_touchdowns >= 10
    assert result.completed_strides >= 5
    assert [event.side for event in events[:4]] == ["right", "left", "right", "left"]


def test_feet_that_never_leave_the_floor_are_not_walking():
    sites = np.zeros((600, 7, 3), dtype=float)
    sites[:, (3, 6), 2] = 0.045
    root = np.column_stack((np.linspace(0, 1, 600), np.zeros(600)))
    try:
        evaluate_behavioral_walk(sites, root, 50.0)
    except ValueError as error:
        assert "접지 이벤트" in str(error)
    else:
        raise AssertionError("바닥에서 미끄러지는 양발을 보행으로 인정했습니다.")

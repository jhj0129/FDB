import numpy as np

from fdb.challenge.dynamic_gait_baseline import pd_action


def test_pd_action_tracks_and_clips_to_motor_limits() -> None:
    action = pd_action(
        np.asarray([0.0, 1.0]), np.zeros(2),
        np.asarray([2.0, 0.0]), np.zeros(2), 100.0, 10.0,
        np.asarray([[-50.0, 50.0], [-20.0, 20.0]]),
    )
    np.testing.assert_allclose(action, [50.0, -20.0])


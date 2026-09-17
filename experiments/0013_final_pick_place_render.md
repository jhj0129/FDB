# Experiment 0013: final integrated Panda demonstration

- Status: completed and independently verified from the rendered trajectory.
- Artifact: `artifacts/fdb_final_pick_place.mp4` (640×360, 30 fps, 21.2 s).
- Selected future: `place_high` from three successful place-height candidates.
- Lift height: 0.14665 m.
- Final target-center error: 0.00558 m.
- Final state: stable, released, and at table height.
- Supporting files: `artifacts/fdb_final_pick_place.json` and
  `artifacts/fdb_final_pick_place.png`.

The renderer reruns candidate selection and then records the chosen physical trajectory;
it refuses to retain a result unless the filmed run itself passes the same objective
lift, placement, release, and stability gates as the underlying skill.


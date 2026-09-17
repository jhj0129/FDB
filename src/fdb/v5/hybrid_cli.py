from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from fdb.v1.models import PhysicsScenario
from fdb.v1.planner import PushCandidatePlanner
from .hybrid import SafeHybridPushPlanner


def main() -> None:
    planner = SafeHybridPushPlanner(Path("models/v5/push_dynamics_ensemble.npz"))
    plans = PushCandidatePlanner().create_candidates()
    nominal = planner.decide(plans)
    ood = planner.decide(
        plans,
        PhysicsScenario(
            scenario_id="의도적_OOD",
            mass_scale=2.3,
            sliding_friction=1.75,
            initial_offset_x=0.22,
            initial_offset_y=-0.2,
        ),
    )
    print(json.dumps({"nominal": asdict(nominal), "ood": asdict(ood)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

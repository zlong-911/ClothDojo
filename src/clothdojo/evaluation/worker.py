"""Internal node process for a disjoint formal-evaluation trajectory shard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from clothdojo.rollout.trajectory_collection import (
    TrajectoryAssetJob,
    TrajectoryCollectionSpec,
    collect_trajectories,
)
from clothdojo.runtime_identity import runtime_source_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    arguments = parser.parse_args()
    plan_path = Path(arguments.plan).expanduser().resolve()
    value = json.loads(plan_path.read_text(encoding="utf-8"))
    if value.get("schema") != "clothdojo_evaluation_node_plan":
        raise ValueError("unsupported evaluation node plan")
    if value.get("runtime_sha256") != runtime_source_sha256():
        raise ValueError("evaluation worker ClothDojo runtime differs from its plan")
    jobs = [
        TrajectoryAssetJob(
            **{
                **item,
                "replicates": tuple(item["replicates"]),
            }
        )
        for item in value["jobs"]
    ]
    spec = TrajectoryCollectionSpec(**value["spec"])
    gpus = tuple(value["gpus"])
    endpoints = tuple(value["policy_endpoints"])
    if len(gpus) != len(endpoints):
        raise ValueError("evaluation node plan GPU/endpoint count differs")
    collect_trajectories(jobs, gpus=gpus, policy_endpoints=endpoints, spec=spec)


if __name__ == "__main__":
    main()

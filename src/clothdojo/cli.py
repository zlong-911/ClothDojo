"""ClothDojo command-line interface."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> None:
    parser = _parser()
    args, passthrough = parser.parse_known_args(argv)
    if args.command == "env" and args.env_command == "check":
        if passthrough:
            parser.error(f"unrecognized arguments: {' '.join(passthrough)}")
        from clothdojo.diagnostics import inspect_environment

        report = inspect_environment(
            include_training=args.training,
            smoke=args.smoke,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        if not report["compatible"]:
            raise SystemExit(1)
        return
    if args.command == "train":
        if args.check:
            from clothdojo.training.rlinf import check as check_rlinf_training

            report = check_rlinf_training(
                kind=args.training_kind,
                config_path=args.config,
                overrides=passthrough,
            )
            print(json.dumps(report, indent=2, sort_keys=True))
            return
        from clothdojo.training.rlinf import run as run_rlinf_training

        run_rlinf_training(
            kind=args.training_kind,
            config_path=args.config,
            overrides=passthrough,
        )
        return
    if args.command == "data" and args.data_command == "export-lerobot":
        if passthrough:
            parser.error(f"unrecognized arguments: {' '.join(passthrough)}")
        from clothdojo.data.lerobot import export_review_dataset, inspect_review_export

        if args.plan_only:
            plan, sources = inspect_review_export(args.source)
            summary = plan.to_json_dict()
            summary.pop("episodes")
            summary.pop("source_action_counts")
            reference = sources[sorted(sources)[0]]
            summary.update(
                {
                    "review_exports": len(sources),
                    "review_format": "LeRobot v3.0",
                    "output_format": "LeRobot v2.1",
                    "camera_names": [
                        key.removeprefix("observation.images.") for key in reference.video_keys
                    ],
                }
            )
            print(json.dumps(summary, indent=2, sort_keys=True))
            return
        if args.output is None:
            parser.error("--output is required unless --plan-only is used")
        result = export_review_dataset(
            args.source,
            args.output,
            repo_id=args.repo_id,
            image_writer_threads=args.image_writer_threads,
        )
        print(json.dumps(result.to_json_dict(), indent=2, sort_keys=True))
        return
    if args.command == "data" and args.data_command == "compute-norm-stats":
        if passthrough:
            parser.error(f"unrecognized arguments: {' '.join(passthrough)}")
        from clothdojo.training.openpi import compute_norm_stats

        result = compute_norm_stats(
            args.dataset,
            args.output,
            config_name=args.config_name,
            block_size=args.block_size,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.command == "data" and args.data_command == "check-cfg":
        from clothdojo.training.cfg import inspect_cfg_training

        result = inspect_cfg_training(args.config, passthrough)
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if passthrough:
        parser.error(f"unrecognized arguments: {' '.join(passthrough)}")
    if args.command == "rollout" and args.rollout_command == "check-rtc":
        from clothdojo.rollout.check import run_rtc_rollout_check

        report = run_rtc_rollout_check(
            args.scenario,
            args.assets,
            args.trajectory_output,
            args.lerobot_output,
            device=args.device,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        if not report["compatible"]:
            raise SystemExit(1)
        return
    if args.command == "rollout" and args.rollout_command == "make-states":
        from clothdojo.rollout.initial_states import generate_initial_states

        report = generate_initial_states(
            args.assets,
            args.output,
            garment_id=args.garment,
            state_count=args.count,
            candidate_batch=args.batch_size,
            max_candidates=args.max_candidates,
            seed=args.seed,
            hold_seconds=args.hold_seconds,
            settle_seconds=args.settle_seconds,
            device=args.device,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    if args.command == "sim" and args.sim_command == "check":
        from clothdojo.sim.check import run_shirt_flatten_check

        report = run_shirt_flatten_check(
            args.assets,
            device=args.device,
            world_count=args.worlds,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        if not report["compatible"]:
            raise SystemExit(1)
        return
    parser.error("a command is required")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="clothdojo")
    commands = parser.add_subparsers(dest="command")

    environment = commands.add_parser("env", help="inspect the project environment")
    environment_commands = environment.add_subparsers(dest="env_command")
    check = environment_commands.add_parser("check", help="validate runtime versions")
    check.add_argument(
        "--training",
        action="store_true",
        help="also require the optional RLINF training stack",
    )
    check.add_argument(
        "--smoke",
        action="store_true",
        help="run a deterministic Newton build/collide/step smoke test on CPU",
    )

    simulation = commands.add_parser("sim", help="run Newton scene checks")
    simulation_commands = simulation.add_subparsers(dest="sim_command")
    simulation_check = simulation_commands.add_parser(
        "check", help="validate the migrated shirt-flattening slice"
    )
    simulation_check.add_argument("--assets", default="assets")
    simulation_check.add_argument("--device", default="cuda:0")
    simulation_check.add_argument("--worlds", type=int, default=1)

    rollout = commands.add_parser("rollout", help="run policy rollout acceptance gates")
    rollout_commands = rollout.add_subparsers(dest="rollout_command")
    rtc_check = rollout_commands.add_parser(
        "check-rtc",
        help="run one scripted RTC episode through Newton and trajectory I/O",
    )
    rtc_check.add_argument("--scenario", required=True)
    rtc_check.add_argument("--assets", default="assets")
    rtc_check.add_argument("--trajectory-output", required=True)
    rtc_check.add_argument("--lerobot-output", required=True)
    rtc_check.add_argument("--device", default="cuda:0")
    make_states = rollout_commands.add_parser(
        "make-states",
        help="generate a validated cloth initial-state pool",
    )
    make_states.add_argument("--garment", default="rand_ZZCWITH5HY")
    make_states.add_argument("--count", type=int, default=32)
    make_states.add_argument("--output", required=True)
    make_states.add_argument("--assets", default="assets")
    make_states.add_argument("--device", default="cuda:0")
    make_states.add_argument("--batch-size", type=int, default=8)
    make_states.add_argument("--max-candidates", type=int, default=512)
    make_states.add_argument("--seed", type=int, default=260815)
    make_states.add_argument("--hold-seconds", type=float, default=1.5)
    make_states.add_argument("--settle-seconds", type=float, default=2.5)

    train = commands.add_parser("train", help="run a ClothDojo config through RLINF")
    train.add_argument("training_kind", choices=("pi05", "steam", "cfg"))
    train.add_argument("--config", required=True, help="path to the root Hydra YAML config")
    train.add_argument(
        "--check",
        action="store_true",
        help="compose and validate the config without launching workers",
    )

    data = commands.add_parser("data", help="inspect and export offline datasets")
    data_commands = data.add_subparsers(dest="data_command")
    export = data_commands.add_parser(
        "export-lerobot",
        help="remove bad ranges and export review data as LeRobot v2.1",
    )
    export.add_argument("--source", required=True, help="source dataset root")
    export.add_argument("--output", help="new LeRobot dataset directory")
    export.add_argument("--repo-id", help="LeRobot repository identity stored in metadata")
    export.add_argument(
        "--image-writer-threads",
        type=int,
        default=6,
        help="threads used by the pinned LeRobot image writer",
    )
    export.add_argument(
        "--plan-only",
        action="store_true",
        help="synchronize and validate inputs without decoding or writing data",
    )
    norm_stats = data_commands.add_parser(
        "compute-norm-stats",
        help="compute OpenPI stats using the ALOHA transforms used by training",
    )
    norm_stats.add_argument("--dataset", required=True, help="good-only LeRobot root")
    norm_stats.add_argument("--output", required=True, help="new external stats directory")
    norm_stats.add_argument("--config-name", default="pi05_aloha_robotwin")
    norm_stats.add_argument("--block-size", type=int, default=4096)
    cfg = data_commands.add_parser(
        "check-cfg",
        help="validate labeled expert/rollout datasets referenced by a CFG config",
    )
    cfg.add_argument("--config", required=True, help="path to the CFG Hydra YAML config")
    return parser

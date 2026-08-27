"""ClothDojo command-line interface."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path


def _training_config_schema(path: str | Path) -> str:
    import yaml

    source = Path(path).expanduser().resolve()
    try:
        value = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeError(f"cannot read training config {source}: {exc}") from exc
    if not isinstance(value, dict) or not isinstance(value.get("schema"), str):
        return ""
    return value["schema"]


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
    if args.command == "steam" and args.steam_command == "merge":
        if passthrough:
            parser.error(f"unrecognized arguments: {' '.join(passthrough)}")
        from clothdojo.training.steam import merge_steam_members

        output = merge_steam_members(
            args.member,
            args.output,
            overwrite=args.overwrite,
        )
        print(json.dumps({"checkpoint": str(output)}, indent=2, sort_keys=True))
        return
    if args.command == "train":
        if args.training_kind == "steam" and args.member_index is None:
            parser.error("STEAM training requires --member-index")
        if args.training_kind != "steam" and args.member_index is not None:
            parser.error("--member-index is only valid for STEAM training")
        if args.advantages_config is not None and args.training_kind != "cfg":
            parser.error("--advantages-config is only valid for CFG training")
        if args.training_kind == "cfg" and args.advantages_config is not None:
            if args.gate is not None:
                parser.error("--gate is only valid for openpi-jax training")
            from clothdojo.training.cfg_pipeline import (
                inspect_cfg_pipeline,
                run_cfg_pipeline,
            )

            result = (
                inspect_cfg_pipeline(
                    advantage_config=args.advantages_config,
                    cfg_config=args.config,
                    cfg_overrides=passthrough,
                )
                if args.check
                else run_cfg_pipeline(
                    advantage_config=args.advantages_config,
                    cfg_config=args.config,
                    cfg_overrides=passthrough,
                )
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            return
        if args.training_kind == "openpi-jax":
            if _training_config_schema(args.config) == "clothdojo_pi05_jax_matrix":
                if passthrough:
                    parser.error(f"unrecognized arguments: {' '.join(passthrough)}")
                if args.gate is not None:
                    parser.error("an OpenPI/JAX matrix does not accept --gate")
                from clothdojo.training.matrix import (
                    compose_jax_sft_matrix,
                    inspect_jax_sft_matrix,
                    run_jax_sft_matrix,
                )

                matrix = compose_jax_sft_matrix(args.config)
                result = (
                    inspect_jax_sft_matrix(matrix, inspect_artifacts=True)
                    if args.check
                    else run_jax_sft_matrix(matrix)
                )
                print(json.dumps(result, indent=2, sort_keys=True))
                return
            from clothdojo.training.jax_sft import (
                compose_jax_sft_spec,
                inspect_jax_sft,
                run_jax_sft,
            )

            spec = compose_jax_sft_spec(args.config, passthrough)
            if args.check:
                print(json.dumps(inspect_jax_sft(spec), indent=2, sort_keys=True))
                return
            if args.gate is None:
                parser.error("openpi-jax training requires --gate")
            result = run_jax_sft(spec, args.gate)
            print(json.dumps(result, indent=2, sort_keys=True))
            return
        if args.gate is not None:
            parser.error("--gate is only valid for openpi-jax training")
        if args.check:
            from clothdojo.training.rlinf import check as check_rlinf_training

            report = check_rlinf_training(
                kind=args.training_kind,
                config_path=args.config,
                overrides=passthrough,
                steam_member_index=args.member_index,
            )
            print(json.dumps(report, indent=2, sort_keys=True))
            return
        from clothdojo.training.rlinf import run as run_rlinf_training

        run_rlinf_training(
            kind=args.training_kind,
            config_path=args.config,
            overrides=passthrough,
            steam_member_index=args.member_index,
        )
        return
    if args.command == "policy" and args.policy_command == "serve-sft":
        if passthrough:
            parser.error(f"unrecognized arguments: {' '.join(passthrough)}")
        from clothdojo.rollout.policy import serve_sft_policy

        serve_sft_policy(
            args.checkpoint,
            args.norm_stats,
            openpi_config=args.openpi_config,
            action_horizon=args.action_horizon,
            num_steps=args.num_steps,
            autocast=not args.no_autocast,
            rtc_execution_horizon=args.rtc_execution_horizon,
            rtc_max_guidance_weight=args.rtc_max_guidance_weight,
            rtc_prefix_schedule=args.rtc_prefix_schedule,
            device=args.device,
            host=args.host,
            port=args.port,
        )
        return
    if args.command == "data" and args.data_command == "export-lerobot":
        if passthrough:
            parser.error(f"unrecognized arguments: {' '.join(passthrough)}")
        source_root = Path(args.source).expanduser()
        source_manifest = source_root / "meta.json"
        if not source_manifest.is_file():
            source_manifest = source_root / "manifest.json"
        source_schema = None
        if source_manifest.is_file():
            try:
                source_schema = json.loads(source_manifest.read_text(encoding="utf-8")).get(
                    "schema"
                )
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"cannot read source manifest: {source_manifest}: {exc}"
                ) from exc
        if source_schema == "clothdojo_rollout":
            from clothdojo.data.rollout import (
                export_rollout_collection,
                inspect_rollout_collection_export,
            )

            if args.plan_only:
                print(
                    json.dumps(
                        inspect_rollout_collection_export(args.source),
                        indent=2,
                        sort_keys=True,
                    )
                )
                return
            output = args.output or str(source_root / "lerobot")
            result = export_rollout_collection(
                args.source,
                output,
                assets_root=args.assets,
                device=args.device,
                repo_id=args.repo_id,
                image_writer_threads=args.image_writer_threads,
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            return
        from clothdojo.data.lerobot import (
            export_review_dataset,
            inspect_review_export,
            republish_review_dataset,
        )

        if args.plan_only:
            plan, sources = inspect_review_export(
                args.source,
                min_segment_frames=args.min_segment_frames,
            )
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
        if args.reuse_from is None:
            result = export_review_dataset(
                args.source,
                args.output,
                repo_id=args.repo_id,
                image_writer_threads=args.image_writer_threads,
                min_segment_frames=args.min_segment_frames,
            )
        else:
            result = republish_review_dataset(
                args.source,
                args.reuse_from,
                args.output,
                repo_id=args.repo_id,
                min_segment_frames=args.min_segment_frames,
            )
        print(json.dumps(result.to_json_dict(), indent=2, sort_keys=True))
        return
    if args.command == "data" and args.data_command == "import-legacy-collection":
        if passthrough:
            parser.error(f"unrecognized arguments: {' '.join(passthrough)}")
        from clothdojo.data.legacy_collection import import_legacy_collection

        last_reported = -1

        def report(files: int, total_files: int, copied: int, total_bytes: int) -> None:
            nonlocal last_reported
            gib = copied // (1024**3)
            if gib != last_reported or files == total_files:
                last_reported = gib
                print(
                    f"imported {files}/{total_files} files "
                    f"({copied / 1024**3:.1f}/{total_bytes / 1024**3:.1f} GiB)",
                    flush=True,
                )

        result = import_legacy_collection(
            args.source,
            args.output,
            include_reviews=not args.without_reviews,
            workers=args.workers,
            progress=report,
        )
        print(json.dumps(result.to_json_dict(), indent=2, sort_keys=True))
        return
    if args.command == "data" and args.data_command == "migrate-v3-indexed-view":
        if passthrough:
            parser.error(f"unrecognized arguments: {' '.join(passthrough)}")
        from clothdojo.data.v3_indexed_view import (
            compose_indexed_view_spec,
            inspect_indexed_view,
            publish_indexed_view,
        )

        spec = compose_indexed_view_spec(args.config)
        result = inspect_indexed_view(spec) if args.check else publish_indexed_view(spec)
        print(json.dumps(result, indent=2, sort_keys=True))
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
    if args.command == "collect" and args.collect_command == "serve":
        if passthrough:
            parser.error(f"unrecognized arguments: {' '.join(passthrough)}")
        from clothdojo.collection.web import serve

        serve(args.config)
        return
    if args.command == "benchmark" and args.benchmark_command == "check-protocol":
        if passthrough:
            parser.error(f"unrecognized arguments: {' '.join(passthrough)}")
        from clothdojo.evaluation.protocol import load_evaluation_protocol

        protocol = load_evaluation_protocol(args.protocol)
        result = protocol.to_json_dict()
        result.update(
            {
                "fingerprint": protocol.fingerprint,
                "horizon_steps": protocol.horizon_steps,
                "metric_stride": protocol.metric_stride,
                "stable_samples": protocol.stable_samples,
            }
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.command == "benchmark" and args.benchmark_command == "run":
        if passthrough:
            parser.error(f"unrecognized arguments: {' '.join(passthrough)}")
        from clothdojo.evaluation.collection import run_evaluation_condition

        result = run_evaluation_condition(
            args.scenario,
            args.protocol,
            args.states,
            args.templates,
            _comma_separated(args.policy, "policy"),
            args.output,
            condition_id=args.condition_id,
            expected_checkpoint_sha256=args.expected_checkpoint_sha256,
            assets_root=args.assets,
            gpus=_comma_separated(args.gpus, "gpus"),
            limit_per_asset=args.limit_per_asset,
            seen_assets=args.seen_assets,
            unseen_assets=args.unseen_assets,
            execution=args.execution,
            exec_horizon=args.exec_horizon,
            batch_size=args.batch_size,
            replicates=args.replicates,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.command == "benchmark" and args.benchmark_command == "make-templates":
        if passthrough:
            parser.error(f"unrecognized arguments: {' '.join(passthrough)}")
        from clothdojo.evaluation.templates import generate_template_collection

        result = generate_template_collection(
            args.states,
            args.protocol,
            args.output,
            assets_root=args.assets,
            gpus=_comma_separated(args.gpus, "gpus"),
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.command == "benchmark" and args.benchmark_command == "summarize":
        if passthrough:
            parser.error(f"unrecognized arguments: {' '.join(passthrough)}")
        from clothdojo.evaluation.results import summarize_paired_conditions

        result = summarize_paired_conditions(args.condition)
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if passthrough:
        parser.error(f"unrecognized arguments: {' '.join(passthrough)}")
    if args.command == "rollout" and args.rollout_command == "make-states":
        from clothdojo.rollout.initial_states import generate_initial_state_collection

        report = generate_initial_state_collection(
            args.assets,
            args.output,
            expert_trajectories=args.expert_trajectories,
            seen_assets=args.seen_assets,
            unseen_assets=args.unseen_assets,
            states_per_asset=args.states_per_asset,
            gpus=_comma_separated(args.gpus, "gpus"),
            candidate_batch=args.batch_size,
            max_candidates=args.max_candidates,
            seed=args.seed,
            drop_seconds=args.drop_seconds,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    if args.command == "rollout" and args.rollout_command == "run":
        from clothdojo.rollout.collection import run_rollout_collection

        report = run_rollout_collection(
            args.scenario,
            args.states,
            _comma_separated(args.policy, "policy"),
            args.output,
            expected_checkpoint_sha256=args.expected_checkpoint_sha256,
            assets_root=args.assets,
            gpus=_comma_separated(args.gpus, "gpus"),
            limit_per_asset=args.limit_per_asset,
            seen_assets=args.seen_assets,
            unseen_assets=args.unseen_assets,
            execution=args.execution,
            exec_horizon=args.exec_horizon,
            batch_size=args.batch_size,
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
    from clothdojo.data.annotations import DEFAULT_MIN_SEGMENT_FRAMES

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

    rollout = commands.add_parser("rollout", help="generate initial states and policy trajectories")
    rollout_commands = rollout.add_subparsers(dest="rollout_command")
    make_states = rollout_commands.add_parser(
        "make-states",
        help="select garments and generate one validated state pool per asset",
    )
    make_states.add_argument("--seen-assets", type=int, default=0)
    make_states.add_argument("--unseen-assets", type=int, required=True)
    make_states.add_argument("--states-per-asset", type=int, default=32)
    make_states.add_argument(
        "--expert-trajectories",
        default=os.environ.get("CLOTHDOJO_EXPERT_TRAJECTORIES"),
        help="expert trajectory or LeRobot dataset used to identify seen assets",
    )
    make_states.add_argument("--output", required=True)
    make_states.add_argument("--assets", default="assets")
    make_states.add_argument("--gpus", default="0", help="comma-separated simulation GPU ids")
    make_states.add_argument("--batch-size", type=int, default=8)
    make_states.add_argument("--max-candidates", type=int, default=512)
    make_states.add_argument("--seed", type=int, default=260815)
    make_states.add_argument("--drop-seconds", type=float, default=3.0)
    run_rollout = rollout_commands.add_parser(
        "run",
        help="run an SFT policy over an initial-state pool",
    )
    run_rollout.add_argument("--scenario", required=True)
    run_rollout.add_argument("--states", required=True)
    run_rollout.add_argument(
        "--policy",
        default="ws://127.0.0.1:8080",
        help="one endpoint or a comma-separated endpoint pool",
    )
    run_rollout.add_argument("--expected-checkpoint-sha256")
    run_rollout.add_argument("--output", required=True)
    run_rollout.add_argument("--assets", default="assets")
    run_rollout.add_argument("--gpus", default="0", help="comma-separated simulation GPU ids")
    run_rollout.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Newton worlds rendered and inferred together per GPU (default: 4)",
    )
    run_rollout.add_argument("--limit-per-asset", type=int)
    run_rollout.add_argument("--seen-assets", type=int)
    run_rollout.add_argument("--unseen-assets", type=int)
    run_rollout.add_argument(
        "--execution",
        choices=("chunk", "rtc"),
        default="chunk",
        help="independent chunks or synchronous guided RTC (default: chunk)",
    )
    run_rollout.add_argument(
        "--exec-horizon",
        type=int,
        default=10,
        help="actions executed before replanning (default: 10)",
    )

    policy = commands.add_parser("policy", help="serve frozen rollout policies")
    policy_commands = policy.add_subparsers(dest="policy_command")
    serve_sft = policy_commands.add_parser(
        "serve-sft",
        help="serve an RLINF SFT checkpoint with RTC support",
    )
    serve_sft.add_argument("--checkpoint", required=True)
    serve_sft.add_argument("--norm-stats", required=True)
    serve_sft.add_argument("--openpi-config", default="pi05_aloha_robotwin")
    serve_sft.add_argument("--action-horizon", type=int, default=50)
    serve_sft.add_argument("--num-steps", type=int, default=10)
    serve_sft.add_argument("--rtc-execution-horizon", type=int, default=10)
    serve_sft.add_argument("--rtc-max-guidance-weight", type=float, default=10.0)
    serve_sft.add_argument(
        "--rtc-prefix-schedule",
        choices=("linear", "exp", "ones", "zeros"),
        default="linear",
    )
    serve_sft.add_argument(
        "--no-autocast",
        action="store_true",
        help="disable CUDA BF16 autocast for a full-FP32 inference baseline",
    )
    serve_sft.add_argument("--device", default="cuda:0")
    serve_sft.add_argument("--host", default="127.0.0.1")
    serve_sft.add_argument("--port", type=int, default=8080)

    benchmark = commands.add_parser(
        "benchmark", help="validate protocols and summarize paired evaluations"
    )
    benchmark_commands = benchmark.add_subparsers(dest="benchmark_command")
    benchmark_check = benchmark_commands.add_parser(
        "check-protocol", help="validate and fingerprint one evaluation protocol"
    )
    benchmark_check.add_argument("--protocol", required=True)
    benchmark_template = benchmark_commands.add_parser(
        "make-templates",
        help="free-settle one canonical target for every initial-state asset",
    )
    benchmark_template.add_argument("--states", required=True)
    benchmark_template.add_argument("--protocol", required=True)
    benchmark_template.add_argument("--output", required=True)
    benchmark_template.add_argument("--assets", default="assets")
    benchmark_template.add_argument("--gpus", default="0")
    benchmark_run = benchmark_commands.add_parser(
        "run", help="evaluate one checkpoint/execution condition"
    )
    benchmark_run.add_argument("--condition-id", required=True)
    benchmark_run.add_argument("--scenario", required=True)
    benchmark_run.add_argument("--protocol", required=True)
    benchmark_run.add_argument("--states", required=True)
    benchmark_run.add_argument("--templates", required=True)
    benchmark_run.add_argument(
        "--policy",
        default="ws://127.0.0.1:8080",
        help="one endpoint or a comma-separated endpoint pool",
    )
    benchmark_run.add_argument("--expected-checkpoint-sha256")
    benchmark_run.add_argument("--output", required=True)
    benchmark_run.add_argument("--assets", default="assets")
    benchmark_run.add_argument("--gpus", default="0")
    benchmark_run.add_argument("--batch-size", type=int, default=4)
    benchmark_run.add_argument("--replicates", type=int, default=1)
    benchmark_run.add_argument("--limit-per-asset", type=int)
    benchmark_run.add_argument("--seen-assets", type=int)
    benchmark_run.add_argument("--unseen-assets", type=int)
    benchmark_run.add_argument(
        "--execution", choices=("chunk", "rtc"), default="chunk"
    )
    benchmark_run.add_argument("--exec-horizon", type=int, default=10)
    benchmark_summary = benchmark_commands.add_parser(
        "summarize", help="compare completed conditions over identical cases"
    )
    benchmark_summary.add_argument(
        "--condition",
        action="append",
        required=True,
        help="completed condition directory; repeat in baseline-first order",
    )

    train = commands.add_parser("train", help="run a guarded ClothDojo training workflow")
    train.add_argument(
        "training_kind", choices=("openpi-jax", "openpi-rlinf", "steam", "cfg")
    )
    train.add_argument("--config", required=True, help="path to the root Hydra YAML config")
    train.add_argument(
        "--check",
        action="store_true",
        help="compose and validate the config without launching workers",
    )
    train.add_argument(
        "--gate",
        choices=("initialize", "overfit", "pilot", "parity", "full"),
        help="guarded OpenPI/JAX stage; full requires all four acceptance gates",
    )
    train.add_argument(
        "--advantages-config",
        help="project-owned STEAM labeling config to run before CFG",
    )
    train.add_argument(
        "--member-index",
        type=int,
        choices=(0, 1, 2),
        help="independent STEAM ensemble member to train",
    )

    steam = commands.add_parser("steam", help="assemble STEAM evaluator checkpoints")
    steam_commands = steam.add_subparsers(dest="steam_command")
    steam_merge = steam_commands.add_parser(
        "merge", help="merge three independently trained evaluator members"
    )
    steam_merge.add_argument(
        "--member",
        action="append",
        required=True,
        help="member actor checkpoint; repeat exactly three times",
    )
    steam_merge.add_argument("--output", required=True)
    steam_merge.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing output directory",
    )

    data = commands.add_parser("data", help="inspect and export offline datasets")
    data_commands = data.add_subparsers(dest="data_command")
    export = data_commands.add_parser(
        "export-lerobot",
        help="remove bad ranges and export review data as LeRobot v2.1",
    )
    export.add_argument("--source", required=True, help="source dataset root")
    export.add_argument("--output", help="new LeRobot dataset directory")
    export.add_argument("--assets", default="assets", help="assets used to replay rollout states")
    export.add_argument("--device", default="cuda:0", help="render device for rollout replay")
    export.add_argument("--repo-id", help="LeRobot repository identity stored in metadata")
    export.add_argument(
        "--reuse-from",
        help="completed ClothDojo v2.1 superset whose immutable videos can be reused",
    )
    export.add_argument(
        "--image-writer-threads",
        type=int,
        default=6,
        help="threads used by the pinned LeRobot image writer",
    )
    export.add_argument(
        "--min-segment-frames",
        type=int,
        default=DEFAULT_MIN_SEGMENT_FRAMES,
        help="drop reviewed-good fragments shorter than this many frames",
    )
    export.add_argument(
        "--plan-only",
        action="store_true",
        help="synchronize and validate inputs without decoding or writing data",
    )
    legacy_import = data_commands.add_parser(
        "import-legacy-collection",
        help="copy and normalize one validated pre-ClothDojo collection",
    )
    legacy_import.add_argument("--source", required=True, help="legacy dataset root")
    legacy_import.add_argument("--output", required=True, help="new ClothDojo dataset root")
    legacy_import.add_argument("--workers", type=int, default=8)
    legacy_import.add_argument(
        "--without-reviews",
        action="store_true",
        help="migrate only canonical trajectories and annotations",
    )
    indexed_view = data_commands.add_parser(
        "migrate-v3-indexed-view",
        help="publish immutable v3 media as a project-owned LeRobot v2.1 indexed view",
    )
    indexed_view.add_argument("--config", required=True)
    indexed_view.add_argument("--check", action="store_true")
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

    collect = commands.add_parser(
        "collect",
        help="run teleoperation, review, HIL and asset-curation workflows",
    )
    collect_commands = collect.add_subparsers(dest="collect_command")
    collect_serve = collect_commands.add_parser(
        "serve",
        help="serve the local collection console",
    )
    collect_serve.add_argument(
        "--config",
        default="configs/collection.yaml",
        help="deployment-specific collection configuration",
    )
    return parser


def _comma_separated(value: str, name: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if not items or len(items) != len(set(items)):
        raise ValueError(f"{name} must be a non-empty comma-separated list without duplicates")
    return items

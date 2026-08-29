"""Dynamics-consistent ALOHA action reconstruction for geometric retargeting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import least_squares

from clothdojo.robots import ALOHA2, robot_profile
from clothdojo.sim.grasp import ATTACH_OPENNESS_THRESHOLD
from clothdojo.sim.runtime import ShirtFlattenRuntime
from clothdojo.sim.scene import ShirtFlattenSpec, build_scene

_BASE_MAX_NFEV = 3
_REFINED_MAX_NFEV = 12
_REFINE_ATTACH_POSITION_M = 0.003
_REFINE_APPROACH_RAD = 0.02
_ARM_POSITION_LIMIT_RAD = 0.025
_ARM_VELOCITY_LIMIT_RAD_S = 1.0
_ATTACH_POSITION_LIMIT_M = 0.01
PREGRASP_APPROACH_TOLERANCE_RAD = 0.05
_ACTION_SEED_RESIDUAL_SCALE = 0.03
_WINDOW_TERMINAL_POSITION_SCALE = 0.1
_WINDOW_TERMINAL_VELOCITY_SCALE = 0.01
_TASK_RESIDUAL_SIZE = 14
_ACTION_REPLAY_POSITION_LIMIT_RAD = 1.0e-4
_ACTION_REPLAY_VELOCITY_LIMIT_RAD_S = 1.0e-3
_DEFAULT_ARM_EFFORT_SCALE = 2.0
_WINDOW_HORIZON = 3
_MAX_WINDOW_REPAIRS = 3


@dataclass(frozen=True, slots=True)
class DynamicsRetargetResult:
    joint_q: np.ndarray
    joint_qd: np.ndarray
    body_q: np.ndarray
    body_qd: np.ndarray
    gripper_open_state: np.ndarray
    joint_target: np.ndarray
    gripper_open_command: np.ndarray
    attach_positions: np.ndarray
    attach_errors: np.ndarray
    validation: dict[str, object]


class AlohaDynamicsRetargeter:
    """Turn a geometric reference into replayable target-robot commands."""

    def __init__(
        self,
        assets_root: str | Path,
        *,
        garment_id: str,
        control_fps: float,
        sim_substeps: int,
        device: str,
        target_robot: str = ALOHA2,
        effort_scale: float = _DEFAULT_ARM_EFFORT_SCALE,
    ) -> None:
        rounded_fps = round(float(control_fps))
        if abs(float(control_fps) - rounded_fps) > 1.0e-9:
            raise ValueError("dynamics retarget requires an integer control rate")
        self.profile = robot_profile(target_robot)
        self.control_fps = float(control_fps)
        self.sim_substeps = int(sim_substeps)
        self.arm = np.asarray(self.profile.arm_coord_indices, dtype=np.int64)
        self.arm_dof = np.asarray(self.profile.arm_dof_indices, dtype=np.int64)
        profile_gripper_dof = np.asarray(self.profile.gripper_dof_indices, dtype=np.int64)
        self.arm_count = len(self.arm)
        self.effort_scale = float(effort_scale)
        if not np.isfinite(self.effort_scale) or self.effort_scale < 1.0:
            raise ValueError("effort_scale must be finite and at least 1.0")
        scene_spec = ShirtFlattenSpec(
            garment_id=garment_id,
            control_fps=int(rounded_fps),
            sim_substeps=int(sim_substeps),
        )
        scene = build_scene(
            Path(assets_root),
            spec=scene_spec,
            device=device,
            visuals=False,
            robot_cloth_collision=False,
            robot_id=target_robot,
        )
        controlled_dof = np.concatenate((self.arm_dof, profile_gripper_dof))
        _scale_effort_limits(scene.model, controlled_dof, self.effort_scale)
        self.runtime = ShirtFlattenRuntime(
            scene,
            use_cuda_graph=False,
            rigid_deterministic=False,
        )
        replay_scene = build_scene(
            Path(assets_root),
            spec=scene_spec,
            device=device,
            visuals=False,
            robot_cloth_collision=False,
            robot_id=target_robot,
        )
        _scale_effort_limits(replay_scene.model, controlled_dof, self.effort_scale)
        self.replay_runtime = ShirtFlattenRuntime(
            replay_scene,
            use_cuda_graph=False,
            # Only action-only cold replays are published and compared by the
            # reproducibility gate.  Keeping disposable optimization trials
            # on the fast kernels avoids repeated deterministic module setup.
            rigid_deterministic=True,
        )
        shooting_scene = build_scene(
            Path(assets_root),
            spec=ShirtFlattenSpec(
                garment_id=garment_id,
                world_count=self.arm_count + 3,
                control_fps=int(rounded_fps),
                sim_substeps=int(sim_substeps),
            ),
            device=device,
            visuals=False,
            robot_cloth_collision=False,
            robot_id=target_robot,
        )
        _scale_effort_limits(shooting_scene.model, controlled_dof, self.effort_scale)
        self.shooting_runtime = ShirtFlattenRuntime(
            shooting_scene,
            use_cuda_graph=False,
            # Batched worlds are disposable local probes.  Deterministic
            # MuJoCo-Warp kernels make this O(arm_dofs) search several times
            # slower, while only the single-world action replay is published
            # and compared by the reproducibility gate.
            rigid_deterministic=False,
        )
        self.shooting_world_count = int(shooting_scene.model.world_count)
        self.shooting_world_offsets = np.asarray(shooting_scene.world_offsets, dtype=np.float32)
        self.dt = self.runtime.control_dt
        self.gripper = np.asarray(self.profile.gripper_coord_indices, dtype=np.int64)
        self.gripper_dof = profile_gripper_dof
        self.gripper_leaders = self.gripper[[0, 2]]
        self.gripper_dof_leaders = self.gripper_dof[[0, 2]]
        self.attach_body_indices = tuple(
            self.profile.body_names.index(frame.parent_body) for frame in self.profile.attach_frames
        )
        self.controlled_q = np.concatenate((self.arm, self.gripper))
        self.controlled_dof = np.concatenate((self.arm_dof, self.gripper_dof))
        model = self.runtime.model
        self.lower = np.asarray(model.joint_limit_lower.numpy()[self.arm_dof], dtype=np.float64)
        self.upper = np.asarray(model.joint_limit_upper.numpy()[self.arm_dof], dtype=np.float64)
        self.stiffness = np.asarray(model.joint_target_ke.numpy()[self.arm_dof], dtype=np.float64)
        self.damping = np.asarray(model.joint_target_kd.numpy()[self.arm_dof], dtype=np.float64)
        self.effort_limit = np.asarray(
            model.joint_effort_limit.numpy()[self.arm_dof], dtype=np.float64
        )
        self.gripper_lower = np.asarray(
            model.joint_limit_lower.numpy()[self.gripper_dof_leaders],
            dtype=np.float64,
        )
        self.gripper_upper = np.asarray(
            model.joint_limit_upper.numpy()[self.gripper_dof_leaders],
            dtype=np.float64,
        )
        self.gripper_stiffness = np.asarray(
            model.joint_target_ke.numpy()[self.gripper_dof_leaders],
            dtype=np.float64,
        )
        self.gripper_damping = np.asarray(
            model.joint_target_kd.numpy()[self.gripper_dof_leaders],
            dtype=np.float64,
        )
        self.gripper_effort_limit = np.asarray(
            model.joint_effort_limit.numpy()[self.gripper_dof_leaders],
            dtype=np.float64,
        )
        if (
            np.any(self.lower >= self.upper)
            or np.any(self.stiffness <= 0.0)
            or np.any(self.damping < 0.0)
            or np.any(self.effort_limit <= 0.0)
            or np.any(self.gripper_stiffness <= 0.0)
            or np.any(self.gripper_damping < 0.0)
            or np.any(self.gripper_effort_limit <= 0.0)
        ):
            raise RuntimeError("target robot actuator limits and gains are incomplete")
        self.closed = np.asarray(self.profile.gripper_closed_positions, dtype=np.float64)
        self.opened = np.asarray(self.profile.gripper_open_positions, dtype=np.float64)
        self.gripper_span = float(self.opened[0] - self.closed[0])
        if abs(self.gripper_span) <= 1.0e-8:
            raise RuntimeError("target robot leader gripper range is empty")

        import warp as wp

        self._arm_target = wp.zeros((1, self.arm_count), dtype=wp.float32, device=model.device)
        self._gripper_command = wp.zeros((1, 2), dtype=wp.float32, device=model.device)
        self._shooting_arm_target = wp.zeros(
            (self.shooting_world_count, self.arm_count),
            dtype=wp.float32,
            device=model.device,
        )
        self._shooting_gripper_command = wp.zeros(
            (self.shooting_world_count, 2),
            dtype=wp.float32,
            device=model.device,
        )

    def reconstruct(
        self,
        reference_joint_q: np.ndarray,
        reference_joint_qd: np.ndarray,
        reference_gripper_open: np.ndarray,
        target_attach_positions: np.ndarray,
        target_approach_directions: np.ndarray,
        pregrasp_mask: np.ndarray,
        approach_validation_mask: np.ndarray,
        attached: np.ndarray,
    ) -> DynamicsRetargetResult:
        total_started = perf_counter()
        reference_q = np.asarray(reference_joint_q, dtype=np.float32)
        reference_qd = np.asarray(reference_joint_qd, dtype=np.float32)
        reference_open = np.asarray(reference_gripper_open, dtype=np.float32)
        target_attach = np.asarray(target_attach_positions, dtype=np.float32)
        target_approach = np.asarray(target_approach_directions, dtype=np.float32)
        pregrasp = np.asarray(pregrasp_mask, dtype=bool)
        approach_validation = np.asarray(approach_validation_mask, dtype=bool)
        attached_mask = np.asarray(attached, dtype=np.int32) != 0
        state_count = len(reference_q)
        expected = {
            "reference_joint_q": (state_count, len(self.profile.home_joint_q)),
            "reference_joint_qd": (state_count, self.profile.joint_dof_count),
            "reference_gripper_open": (state_count, 2),
            "target_attach_positions": (state_count, 2, 3),
            "target_approach_directions": (state_count, 2, 3),
            "pregrasp_mask": (state_count, 2),
            "approach_validation_mask": (state_count, 2),
            "attached": (state_count, 2),
        }
        actual = {
            "reference_joint_q": reference_q.shape,
            "reference_joint_qd": reference_qd.shape,
            "reference_gripper_open": reference_open.shape,
            "target_attach_positions": target_attach.shape,
            "target_approach_directions": target_approach.shape,
            "pregrasp_mask": pregrasp.shape,
            "approach_validation_mask": approach_validation.shape,
            "attached": attached_mask.shape,
        }
        if state_count < 2 or actual != expected:
            raise ValueError(f"dynamics retarget input shapes differ: {actual} != {expected}")
        if not all(
            np.isfinite(value).all()
            for value in (
                reference_q,
                reference_qd,
                reference_open,
                target_attach,
                target_approach,
            )
        ):
            raise ValueError("dynamics retarget inputs must be finite")
        approach_norm = np.linalg.norm(target_approach, axis=-1)
        if np.any(np.abs(approach_norm - 1.0) > 1.0e-4):
            raise ValueError("target approach directions must be normalized")

        self.runtime.restore_rigid_state(reference_q[0], reference_qd[0])
        initial = self.runtime.rigid_state()
        states = {name: [initial[name]] for name in initial}
        arm_targets: list[np.ndarray] = []
        gripper_commands: list[np.ndarray] = []
        previous_command = reference_q[0, self.arm].astype(np.float64)
        optimizer_evaluations = 0
        task_refinement_misses = 0
        adaptive_refinements = 0
        effort_saturation_count = 0
        action_value_count = 0
        maximum_required_effort_ratio = 0.0
        gripper_effort_saturation_count = 0
        gripper_action_value_count = 0
        maximum_required_gripper_effort_ratio = 0.0
        inverse_dynamics_seconds = 0.0
        shooting_seconds = 0.0
        trial_evaluations = 0
        batched_jacobian_steps = 0
        batched_jacobian_world_evaluations = 0

        for index in range(state_count - 1):
            current_q = states["joint_q"][-1]
            current_qd = states["joint_qd"][-1]
            full_acceleration = np.zeros(self.profile.joint_dof_count, dtype=np.float64)
            full_acceleration[self.controlled_dof] = _one_step_acceleration(
                reference_q[index, self.controlled_q],
                reference_qd[index, self.controlled_dof],
                reference_q[index + 1, self.controlled_q],
                reference_qd[index + 1, self.controlled_dof],
                dt=self.dt,
            )
            inverse_started = perf_counter()
            feedforward_full = self.runtime.inverse_dynamics(full_acceleration[None])[0]
            feedforward = feedforward_full[self.arm_dof]
            inverse_dynamics_seconds += perf_counter() - inverse_started
            required_effort = (
                feedforward
                + self.stiffness * (reference_q[index, self.arm] - current_q[self.arm])
                + self.damping * (reference_qd[index, self.arm_dof] - current_qd[self.arm_dof])
            )
            applied_effort = np.clip(required_effort, -self.effort_limit, self.effort_limit)
            unconstrained_target = (
                current_q[self.arm]
                + (applied_effort + self.damping * current_qd[self.arm_dof]) / self.stiffness
            )
            inverse_target = np.clip(
                unconstrained_target,
                self.lower,
                self.upper,
            )
            effort_saturated = np.abs(required_effort) > self.effort_limit
            inverse_target = _canonicalize_saturated_target(
                current_q[self.arm],
                current_qd[self.arm_dof],
                reference_q[index + 1, self.arm],
                reference_qd[index + 1, self.arm_dof],
                inverse_target,
                required_effort,
                stiffness=self.stiffness,
                damping=self.damping,
                effort_limit=self.effort_limit,
                lower=self.lower,
                upper=self.upper,
                dt=self.dt,
            )
            propagated = np.clip(
                previous_command + reference_q[index + 1, self.arm] - reference_q[index, self.arm],
                self.lower,
                self.upper,
            )
            gripper_target, gripper_required_effort = _inverse_position_drive_target(
                current_q[self.gripper_leaders],
                current_qd[self.gripper_dof_leaders],
                reference_q[index + 1, self.gripper_leaders],
                reference_qd[index + 1, self.gripper_dof_leaders],
                feedforward_full[self.gripper_dof_leaders],
                stiffness=self.gripper_stiffness,
                damping=self.gripper_damping,
                effort_limit=self.gripper_effort_limit,
                lower=self.gripper_lower,
                upper=self.gripper_upper,
            )
            gripper_target = _canonicalize_saturated_target(
                current_q[self.gripper_leaders],
                current_qd[self.gripper_dof_leaders],
                reference_q[index + 1, self.gripper_leaders],
                reference_qd[index + 1, self.gripper_dof_leaders],
                gripper_target,
                gripper_required_effort,
                stiffness=self.gripper_stiffness,
                damping=self.gripper_damping,
                effort_limit=self.gripper_effort_limit,
                lower=self.gripper_lower,
                upper=self.gripper_upper,
                dt=self.dt,
            )
            grip = np.clip(
                (gripper_target - self.closed[0]) / self.gripper_span,
                0.0,
                1.0,
            )
            desired_open = reference_open[index + 1]
            desired_attach = target_attach[index + 1]
            desired_approach = target_approach[index + 1]
            desired_pregrasp = pregrasp[index + 1]
            arm_candidates = [inverse_target.copy()]
            for free_target in (
                reference_q[index + 1, self.arm],
                propagated,
            ):
                arm_candidates.append(np.asarray(free_target, dtype=np.float64).copy())
            candidates = np.stack(
                [np.concatenate((candidate, grip)) for candidate in arm_candidates]
            )
            branch_state = (current_q.copy(), current_qd.copy())

            shooting_started = perf_counter()
            candidate_residuals = self._batch_trial_residuals(
                branch_state,
                candidates,
                desired_open,
                desired_attach,
                desired_approach,
                desired_pregrasp,
            )
            trial_evaluations += len(candidates)
            initial_slot = int(np.argmin([np.dot(value, value) for value in candidate_residuals]))
            initial_command = candidates[initial_slot].copy()
            initial_residual = candidate_residuals[initial_slot]

            def task_residual(
                values: np.ndarray,
                state=branch_state,
                target_open=desired_open,
                target_attach_position=desired_attach,
                target_approach_direction=desired_approach,
                target_pregrasp=desired_pregrasp,
            ) -> np.ndarray:
                nonlocal trial_evaluations
                trial_evaluations += 1
                predicted = self._trial(state, values)
                return self._residual(
                    predicted["joint_q"],
                    predicted["body_q"],
                    target_open,
                    target_attach_position,
                    target_approach_direction,
                    target_pregrasp,
                )

            command = initial_command.copy()
            best_residual = np.concatenate(
                (initial_residual[:12], np.zeros(self.arm_count, dtype=np.float64))
            )
            action_lower = np.concatenate((self.lower, np.zeros(2)))
            action_upper = np.concatenate((self.upper, np.ones(2)))
            free_action = np.arange(self.arm_count)
            if self._needs_arm_refinement(initial_residual):

                def free_residual(
                    values: np.ndarray,
                    action: np.ndarray = command,
                    free_indices: np.ndarray = free_action,
                    seed: np.ndarray = initial_command,
                ) -> np.ndarray:
                    action[free_indices] = values
                    return np.concatenate(
                        (
                            task_residual(action)[:12],
                            _ACTION_SEED_RESIDUAL_SCALE
                            * (action[: self.arm_count] - seed[: self.arm_count]),
                        )
                    )

                def free_jacobian(
                    values: np.ndarray,
                    action: np.ndarray = command,
                    free_indices: np.ndarray = free_action,
                    upper: np.ndarray = action_upper,
                    state: tuple[np.ndarray, np.ndarray] = branch_state,
                    seed: np.ndarray = initial_command,
                    target_open: np.ndarray = desired_open,
                    target_attach_position: np.ndarray = desired_attach,
                    target_approach_direction: np.ndarray = desired_approach,
                    target_pregrasp: np.ndarray = desired_pregrasp,
                ) -> np.ndarray:
                    nonlocal batched_jacobian_steps
                    nonlocal batched_jacobian_world_evaluations
                    base_action = action.copy()
                    base_action[free_indices] = values
                    commands = np.tile(base_action, (len(free_indices) + 1, 1))
                    steps = 1.0e-3 * np.maximum(1.0, np.abs(values))
                    positive = values + steps <= upper[free_indices]
                    steps[~positive] *= -1.0
                    commands[1:, free_indices] += np.diag(steps)
                    task_residuals = self._batch_trial_residuals(
                        state,
                        commands,
                        target_open,
                        target_attach_position,
                        target_approach_direction,
                        target_pregrasp,
                    )
                    residuals = np.concatenate(
                        (
                            task_residuals[:, :12],
                            _ACTION_SEED_RESIDUAL_SCALE
                            * (commands[:, : self.arm_count] - seed[: self.arm_count]),
                        ),
                        axis=1,
                    )
                    batched_jacobian_steps += 1
                    batched_jacobian_world_evaluations += len(commands)
                    return ((residuals[1:] - residuals[0]) / steps[:, None]).T

                fit = least_squares(
                    free_residual,
                    command[free_action],
                    bounds=(
                        action_lower[free_action],
                        action_upper[free_action],
                    ),
                    jac=free_jacobian,
                    max_nfev=_BASE_MAX_NFEV,
                    x_scale="jac",
                    xtol=1.0e-7,
                    ftol=1.0e-7,
                    gtol=1.0e-7,
                )
                optimizer_evaluations += int(fit.nfev)
                if self._needs_arm_refinement(fit.fun):
                    adaptive_refinements += 1
                    fit = least_squares(
                        free_residual,
                        fit.x,
                        bounds=(
                            action_lower[free_action],
                            action_upper[free_action],
                        ),
                        jac=free_jacobian,
                        max_nfev=_REFINED_MAX_NFEV,
                        x_scale="jac",
                        xtol=1.0e-7,
                        ftol=1.0e-7,
                        gtol=1.0e-7,
                    )
                    optimizer_evaluations += int(fit.nfev)
                if np.dot(fit.fun, fit.fun) < np.dot(best_residual, best_residual):
                    command[free_action] = fit.x
                    best_residual = fit.fun
                else:
                    command[:] = initial_command
                task_refinement_misses += int(self._needs_arm_refinement(best_residual))
            shooting_seconds += perf_counter() - shooting_started

            previous_command = command[: self.arm_count].copy()
            effort_saturation_count += int(np.count_nonzero(effort_saturated))
            action_value_count += len(required_effort)
            maximum_required_effort_ratio = max(
                maximum_required_effort_ratio,
                float(np.max(np.abs(required_effort) / self.effort_limit)),
            )
            gripper_effort_saturation_count += int(
                np.count_nonzero(np.abs(gripper_required_effort) > self.gripper_effort_limit)
            )
            gripper_action_value_count += len(gripper_required_effort)
            maximum_required_gripper_effort_ratio = max(
                maximum_required_gripper_effort_ratio,
                float(
                    np.max(
                        np.abs(gripper_required_effort) / self.gripper_effort_limit
                    )
                ),
            )
            self.runtime.restore_rigid_state(*branch_state)
            self._step(command)
            actual_state = self.runtime.rigid_state()
            for name in states:
                states[name].append(actual_state[name])
            arm_targets.append(command[: self.arm_count].astype(np.float32))
            gripper_commands.append(command[self.arm_count :].astype(np.float32))

        arrays = {name: np.stack(values) for name, values in states.items()}
        action_q = np.stack(arm_targets)
        action_gripper = np.stack(gripper_commands)
        replay_started = perf_counter()
        arrays = self._replay(reference_q[0], reference_qd[0], action_q, action_gripper)
        initial_replay_seconds = perf_counter() - replay_started
        window_started = perf_counter()
        action_q, action_gripper, arrays, window_stats = self._refine_sparse_windows(
            reference_q[0],
            reference_qd[0],
            reference_open,
            target_attach,
            target_approach,
            pregrasp,
            approach_validation,
            action_q,
            action_gripper,
            arrays,
        )
        window_refinement_seconds = perf_counter() - window_started
        optimizer_evaluations += window_stats["optimizer_evaluations"]
        trial_evaluations += window_stats["trial_evaluations"]
        batched_jacobian_steps += window_stats["batched_jacobian_steps"]
        batched_jacobian_world_evaluations += window_stats["batched_jacobian_world_evaluations"]
        replay_started = perf_counter()
        replay = self._replay(reference_q[0], reference_qd[0], action_q, action_gripper)
        validation_replay_seconds = initial_replay_seconds + perf_counter() - replay_started
        replay_position_error = np.abs(
            replay["joint_q"][:, self.arm] - arrays["joint_q"][:, self.arm]
        )
        replay_velocity_error = np.abs(
            replay["joint_qd"][:, self.arm_dof] - arrays["joint_qd"][:, self.arm_dof]
        )
        arrays = replay
        attach_positions = _attach_positions(
            arrays["body_q"], self.profile.attach_frames, self.attach_body_indices
        )
        attach_errors = np.linalg.norm(attach_positions - target_attach, axis=-1)
        active_errors = attach_errors[attached_mask]
        arm_position_error = np.abs(arrays["joint_q"][:, self.arm] - reference_q[:, self.arm])
        arm_velocity_error = np.abs(
            arrays["joint_qd"][:, self.arm_dof] - reference_qd[:, self.arm_dof]
        )
        open_array = np.stack([self._openness(value) for value in arrays["joint_q"]]).astype(
            np.float32
        )
        open_error = np.abs(open_array - reference_open)
        attached_open_error = open_error[attached_mask]
        attach_rising = np.zeros_like(attached_mask)
        attach_rising[0] = attached_mask[0]
        attach_rising[1:] = attached_mask[1:] & ~attached_mask[:-1]
        attach_transition_open = open_array[attach_rising]
        approach_directions = _attach_approach_directions(
            arrays["body_q"], self.profile.attach_frames, self.attach_body_indices
        )
        approach_errors = _direction_angle(approach_directions, target_approach)
        pregrasp_approach_errors = approach_errors[approach_validation]
        metrics = {
            "source_timeline_preserved": True,
            "robot_dynamics_rollout": True,
            "style3d_solver_used": False,
            "actuator_effort_scale": self.effort_scale,
            "state_count": state_count,
            "action_count": state_count - 1,
            "optimizer_evaluations": optimizer_evaluations,
            "trial_evaluations": trial_evaluations,
            "batched_jacobian_steps": batched_jacobian_steps,
            "batched_jacobian_world_evaluations": (batched_jacobian_world_evaluations),
            "task_refinement_misses": task_refinement_misses,
            "adaptive_refinement_count": adaptive_refinements,
            "window_refinement_attempts": window_stats["attempts"],
            "window_refinement_accepts": window_stats["accepts"],
            "max_ik_seed_position_deviation_rad": _max(arm_position_error),
            "max_ik_seed_velocity_deviation_rad_s": _max(arm_velocity_error),
            "max_gripper_openness_error": _max(open_error),
            "max_attached_gripper_openness_error": _max(attached_open_error),
            "max_attach_transition_gripper_openness": _max(attach_transition_open),
            "max_attach_position_error_m": _max(attach_errors),
            "max_attached_position_error_m": _max(active_errors),
            "p95_attached_position_error_m": _quantile(active_errors, 0.95),
            "max_pregrasp_approach_error_rad": _max(pregrasp_approach_errors),
            "p95_pregrasp_approach_error_rad": _quantile(pregrasp_approach_errors, 0.95),
            "effort_saturation_fraction": (effort_saturation_count / action_value_count),
            "maximum_required_effort_ratio": maximum_required_effort_ratio,
            "gripper_effort_saturation_fraction": (
                gripper_effort_saturation_count / gripper_action_value_count
            ),
            "maximum_required_gripper_effort_ratio": maximum_required_gripper_effort_ratio,
            "action_replay_position_error_rad": _max(replay_position_error),
            "action_replay_velocity_error_rad_s": _max(replay_velocity_error),
            "limits": {
                "attach_transition_gripper_openness": ATTACH_OPENNESS_THRESHOLD,
                "attach_position_error_m": _ATTACH_POSITION_LIMIT_M,
                "pregrasp_approach_error_rad": PREGRASP_APPROACH_TOLERANCE_RAD,
                "action_replay_position_error_rad": _ACTION_REPLAY_POSITION_LIMIT_RAD,
                "action_replay_velocity_error_rad_s": (_ACTION_REPLAY_VELOCITY_LIMIT_RAD_S),
            },
            "diagnostic_thresholds": {
                "ik_seed_position_deviation_rad": _ARM_POSITION_LIMIT_RAD,
                "ik_seed_velocity_deviation_rad_s": _ARM_VELOCITY_LIMIT_RAD_S,
            },
            "timing_seconds": {
                "inverse_dynamics": inverse_dynamics_seconds,
                "shooting": shooting_seconds,
                "window_refinement": window_refinement_seconds,
                "validation_replays": validation_replay_seconds,
                "total": perf_counter() - total_started,
            },
        }
        metrics["dynamics_valid"] = bool(
            metrics["max_attach_transition_gripper_openness"] <= ATTACH_OPENNESS_THRESHOLD
            and metrics["max_attach_position_error_m"] <= _ATTACH_POSITION_LIMIT_M
            and metrics["max_pregrasp_approach_error_rad"] <= PREGRASP_APPROACH_TOLERANCE_RAD
            and metrics["action_replay_position_error_rad"] <= _ACTION_REPLAY_POSITION_LIMIT_RAD
            and metrics["action_replay_velocity_error_rad_s"] <= _ACTION_REPLAY_VELOCITY_LIMIT_RAD_S
        )
        return DynamicsRetargetResult(
            joint_q=arrays["joint_q"],
            joint_qd=arrays["joint_qd"],
            body_q=arrays["body_q"],
            body_qd=arrays["body_qd"],
            gripper_open_state=open_array,
            joint_target=action_q,
            gripper_open_command=action_gripper,
            attach_positions=attach_positions.astype(np.float32),
            attach_errors=attach_errors.astype(np.float32),
            validation=metrics,
        )

    def _trial(
        self, state: tuple[np.ndarray, np.ndarray], command: np.ndarray
    ) -> dict[str, np.ndarray]:
        self.runtime.restore_rigid_state(*state)
        self._step(command)
        return self.runtime.rigid_state()

    def _batch_trial_residuals(
        self,
        state: tuple[np.ndarray, np.ndarray],
        commands: np.ndarray,
        desired_open: np.ndarray,
        desired_attach_positions: np.ndarray,
        desired_approach_directions: np.ndarray,
        pregrasp_mask: np.ndarray,
    ) -> np.ndarray:
        values = np.asarray(commands, dtype=np.float32)
        count = len(values)
        if values.shape != (count, self.arm_count + 2):
            raise ValueError("batched shooting commands have an invalid shape")
        if count > self.shooting_world_count:
            raise ValueError("batched shooting exceeds the available physics worlds")
        padded = np.repeat(values[:1], self.shooting_world_count, axis=0)
        padded[:count] = values
        joint_q = np.repeat(
            np.asarray(state[0], dtype=np.float32)[None],
            self.shooting_world_count,
            axis=0,
        )
        joint_qd = np.repeat(
            np.asarray(state[1], dtype=np.float32)[None],
            self.shooting_world_count,
            axis=0,
        )
        self.shooting_runtime.restore_rigid_state_batch(joint_q, joint_qd)
        self._shooting_arm_target.assign(padded[:, : self.arm_count])
        self._shooting_gripper_command.assign(padded[:, self.arm_count :])
        self.shooting_runtime.step_rigid(self._shooting_arm_target, self._shooting_gripper_command)
        predicted = self.shooting_runtime.rigid_state_batch()
        body_q = predicted["body_q"][:count].copy()
        body_q[..., :3] -= self.shooting_world_offsets[:count, None, :]
        return np.stack(
            [
                self._residual(
                    predicted["joint_q"][world],
                    body_q[world],
                    desired_open,
                    desired_attach_positions,
                    desired_approach_directions,
                    pregrasp_mask,
                )
                for world in range(count)
            ]
        )

    def _batch_window_residuals(
        self,
        state: tuple[np.ndarray, np.ndarray],
        command_sequences: np.ndarray,
        desired_open: np.ndarray,
        desired_attach_positions: np.ndarray,
        desired_approach_directions: np.ndarray,
        pregrasp_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        sequences = np.asarray(command_sequences, dtype=np.float32)
        count, horizon, command_count = sequences.shape
        if command_count != self.arm_count + 2:
            raise ValueError("window shooting commands have an invalid shape")
        if count > self.shooting_world_count:
            raise ValueError("window shooting exceeds the available physics worlds")
        if desired_attach_positions.shape[0] != horizon:
            raise ValueError("window shooting targets have an invalid length")

        padded = np.repeat(sequences[:1], self.shooting_world_count, axis=0)
        padded[:count] = sequences
        joint_q = np.repeat(
            np.asarray(state[0], dtype=np.float32)[None],
            self.shooting_world_count,
            axis=0,
        )
        joint_qd = np.repeat(
            np.asarray(state[1], dtype=np.float32)[None],
            self.shooting_world_count,
            axis=0,
        )
        self.shooting_runtime.restore_rigid_state_batch(joint_q, joint_qd)
        step_residuals = []
        for step in range(horizon):
            commands = padded[:, step]
            self._shooting_arm_target.assign(commands[:, : self.arm_count])
            self._shooting_gripper_command.assign(commands[:, self.arm_count :])
            self.shooting_runtime.step_rigid(
                self._shooting_arm_target, self._shooting_gripper_command
            )
            predicted = self.shooting_runtime.rigid_state_batch()
            body_q = predicted["body_q"][:count].copy()
            body_q[..., :3] -= self.shooting_world_offsets[:count, None, :]
            step_residuals.append(
                np.stack(
                    [
                        self._residual(
                            predicted["joint_q"][world],
                            body_q[world],
                            desired_open[step],
                            desired_attach_positions[step],
                            desired_approach_directions[step],
                            pregrasp_mask[step],
                        )
                        for world in range(count)
                    ]
                )
            )
        return (
            np.concatenate(step_residuals, axis=1),
            predicted["joint_q"][:count],
            predicted["joint_qd"][:count],
        )

    def _refine_sparse_windows(
        self,
        initial_q: np.ndarray,
        initial_qd: np.ndarray,
        reference_open: np.ndarray,
        target_attach: np.ndarray,
        target_approach: np.ndarray,
        pregrasp: np.ndarray,
        approach_validation: np.ndarray,
        action_q: np.ndarray,
        action_gripper: np.ndarray,
        arrays: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, int]]:
        refined_actions = action_q.copy()
        refined_gripper = action_gripper.copy()
        attempted: set[tuple[int, int]] = set()
        stats = {
            "attempts": 0,
            "accepts": 0,
            "optimizer_evaluations": 0,
            "trial_evaluations": 0,
            "batched_jacobian_steps": 0,
            "batched_jacobian_world_evaluations": 0,
        }
        action_count = len(refined_actions)
        horizon = min(_WINDOW_HORIZON, action_count)
        if horizon == 0:
            return refined_actions, refined_gripper, arrays, stats

        for _ in range(_MAX_WINDOW_REPAIRS):
            attach_positions = _attach_positions(
                arrays["body_q"], self.profile.attach_frames, self.attach_body_indices
            )
            attach_errors = np.linalg.norm(attach_positions - target_attach, axis=-1)
            approach_directions = _attach_approach_directions(
                arrays["body_q"], self.profile.attach_frames, self.attach_body_indices
            )
            approach_errors = _direction_angle(approach_directions, target_approach)
            selected = _select_repair_window(
                attach_errors,
                approach_errors,
                approach_validation,
                action_count=action_count,
                horizon=horizon,
                attempted=attempted,
            )
            if selected is None:
                break
            start, end, failing_state = selected
            attempted.add((start, end))
            stats["attempts"] += 1

            desired_open = reference_open[start + 1 : end + 1].copy()
            desired_attach = target_attach[start + 1 : end + 1].copy()
            desired_approach = target_approach[start + 1 : end + 1].copy()
            desired_pregrasp = pregrasp[start + 1 : end + 1].copy()
            if end > failing_state:
                desired_attach[-1] = attach_positions[end]
                desired_approach[-1] = approach_directions[end]
                desired_pregrasp[-1] = True

            initial_sequence = refined_actions[start:end].astype(np.float64)
            fixed_gripper = refined_gripper[start:end].astype(np.float64)
            initial_values = initial_sequence.ravel()
            state = (arrays["joint_q"][start], arrays["joint_qd"][start])
            failing_step = failing_state - start - 1
            objective_steps = np.unique((0, failing_step, horizon - 1))
            terminal_q = arrays["joint_q"][end]
            terminal_qd = arrays["joint_qd"][end]

            def objective_residuals(
                task_values: np.ndarray,
                terminal_positions: np.ndarray,
                terminal_velocities: np.ndarray,
                sequences: np.ndarray,
                steps: np.ndarray = objective_steps,
                seed: np.ndarray = initial_sequence,
                target_terminal_q: np.ndarray = terminal_q,
                target_terminal_qd: np.ndarray = terminal_qd,
            ) -> np.ndarray:
                shaped = task_values.reshape(len(task_values), horizon, _TASK_RESIDUAL_SIZE)
                tracked = shaped[:, steps, :12].reshape(len(task_values), -1)
                seed_residual = _ACTION_SEED_RESIDUAL_SCALE * (sequences - seed[None]).reshape(
                    len(task_values), -1
                )
                terminal_residual = np.concatenate(
                    (
                        _WINDOW_TERMINAL_POSITION_SCALE
                        * (terminal_positions[:, self.arm] - target_terminal_q[self.arm]),
                        _WINDOW_TERMINAL_VELOCITY_SCALE
                        * (terminal_velocities[:, self.arm_dof] - target_terminal_qd[self.arm_dof]),
                    ),
                    axis=1,
                )
                return np.concatenate((tracked, seed_residual, terminal_residual), axis=1)

            def residual(
                values: np.ndarray,
                window_state: tuple[np.ndarray, np.ndarray] = state,
                target_open: np.ndarray = desired_open,
                target_attach_positions: np.ndarray = desired_attach,
                target_approach_directions: np.ndarray = desired_approach,
                target_pregrasp: np.ndarray = desired_pregrasp,
                gripper_sequence: np.ndarray = fixed_gripper,
            ) -> np.ndarray:
                stats["trial_evaluations"] += 1
                arm_sequence = values.reshape(horizon, self.arm_count)
                sequence = np.concatenate((arm_sequence, gripper_sequence), axis=1)
                predicted, terminal_positions, terminal_velocities = self._batch_window_residuals(
                    window_state,
                    sequence[None],
                    target_open,
                    target_attach_positions,
                    target_approach_directions,
                    target_pregrasp,
                )
                return objective_residuals(
                    predicted,
                    terminal_positions,
                    terminal_velocities,
                    arm_sequence[None],
                )[0]

            initial_residual = residual(initial_values)

            def jacobian(
                values: np.ndarray,
                residual_count: int = len(initial_residual),
                window_state: tuple[np.ndarray, np.ndarray] = state,
                target_open: np.ndarray = desired_open,
                target_attach_positions: np.ndarray = desired_attach,
                target_approach_directions: np.ndarray = desired_approach,
                target_pregrasp: np.ndarray = desired_pregrasp,
                gripper_sequence: np.ndarray = fixed_gripper,
            ) -> np.ndarray:
                variable_count = len(values)
                result = np.empty((residual_count, variable_count), dtype=np.float64)
                command_count = self.arm_count
                for offset in range(0, variable_count, command_count):
                    indices = np.arange(offset, min(offset + command_count, variable_count))
                    arm_sequences = np.repeat(
                        values.reshape(horizon, command_count)[None],
                        len(indices) + 1,
                        axis=0,
                    )
                    steps = 1.0e-3 * np.maximum(1.0, np.abs(values[indices]))
                    command_indices = indices % command_count
                    positive = values[indices] + steps <= self.upper[command_indices]
                    steps[~positive] *= -1.0
                    for row, (index, step) in enumerate(zip(indices, steps, strict=True), 1):
                        arm_sequences[row, index // command_count, index % command_count] += step
                    sequences = np.concatenate(
                        (
                            arm_sequences,
                            np.repeat(gripper_sequence[None], len(arm_sequences), axis=0),
                        ),
                        axis=2,
                    )
                    task_values, terminal_positions, terminal_velocities = (
                        self._batch_window_residuals(
                            window_state,
                            sequences,
                            target_open,
                            target_attach_positions,
                            target_approach_directions,
                            target_pregrasp,
                        )
                    )
                    values_residual = objective_residuals(
                        task_values,
                        terminal_positions,
                        terminal_velocities,
                        arm_sequences,
                    )
                    stats["batched_jacobian_steps"] += horizon
                    stats["batched_jacobian_world_evaluations"] += len(sequences) * horizon
                    result[:, indices] = (
                        (values_residual[1:] - values_residual[0]) / steps[:, None]
                    ).T
                return result

            fit = least_squares(
                residual,
                initial_values,
                bounds=(
                    np.tile(self.lower, horizon),
                    np.tile(self.upper, horizon),
                ),
                jac=jacobian,
                max_nfev=_REFINED_MAX_NFEV,
                x_scale="jac",
                xtol=1.0e-7,
                ftol=1.0e-7,
                gtol=1.0e-7,
            )
            stats["optimizer_evaluations"] += int(fit.nfev)
            if np.dot(fit.fun, fit.fun) >= np.dot(initial_residual, initial_residual):
                continue

            candidate_actions = refined_actions.copy()
            candidate_actions[start:end] = fit.x.reshape(horizon, self.arm_count)
            candidate_arrays = self._replay(
                initial_q,
                initial_qd,
                candidate_actions,
                refined_gripper,
                deterministic=False,
            )
            baseline_values = self._window_validation_values(
                arrays,
                target_attach,
                target_approach,
                approach_validation,
            )
            candidate_values = self._window_validation_values(
                candidate_arrays,
                target_attach,
                target_approach,
                approach_validation,
            )
            if not _window_candidate_improves(baseline_values, candidate_values):
                continue
            refined_actions = candidate_actions
            arrays = candidate_arrays
            stats["accepts"] += 1
            attempted.clear()

        return refined_actions, refined_gripper, arrays, stats

    def _window_validation_values(
        self,
        arrays: dict[str, np.ndarray],
        target_attach: np.ndarray,
        target_approach: np.ndarray,
        approach_validation: np.ndarray,
    ) -> tuple[float, float]:
        positions = _attach_positions(
            arrays["body_q"], self.profile.attach_frames, self.attach_body_indices
        )
        position_errors = np.linalg.norm(positions - target_attach, axis=-1)
        approach = _attach_approach_directions(
            arrays["body_q"], self.profile.attach_frames, self.attach_body_indices
        )
        approach_errors = _direction_angle(approach, target_approach)[approach_validation]
        return _max(position_errors), _max(approach_errors)

    def _step(self, command: np.ndarray) -> None:
        values = np.asarray(command, dtype=np.float32)
        self._arm_target.assign(values[None, : self.arm_count])
        self._gripper_command.assign(values[None, self.arm_count :])
        self.runtime.step_rigid(self._arm_target, self._gripper_command)

    def _residual(
        self,
        joint_q: np.ndarray,
        body_q: np.ndarray,
        desired_open: np.ndarray,
        desired_attach_positions: np.ndarray,
        desired_approach_directions: np.ndarray,
        pregrasp_mask: np.ndarray,
    ) -> np.ndarray:
        attach_position = _attach_positions(
            np.asarray(body_q, dtype=np.float32)[None],
            self.profile.attach_frames,
            self.attach_body_indices,
        )[0]
        attach_approach = _attach_approach_directions(
            np.asarray(body_q, dtype=np.float32)[None],
            self.profile.attach_frames,
            self.attach_body_indices,
        )[0]
        return _task_space_residual(
            attach_position,
            self._openness(joint_q),
            attach_approach,
            desired_attach_positions,
            desired_open,
            desired_approach_directions,
            pregrasp_mask,
        )

    def _needs_arm_refinement(self, residual: np.ndarray) -> bool:
        values = np.asarray(residual, dtype=np.float64)[:12]
        attach = values[:6].reshape(2, 3) * _ATTACH_POSITION_LIMIT_M
        approach = values[6:12].reshape(2, 3) * PREGRASP_APPROACH_TOLERANCE_RAD
        return bool(
            np.max(np.linalg.norm(attach, axis=-1), initial=0.0) > _REFINE_ATTACH_POSITION_M
            or np.max(np.linalg.norm(approach, axis=-1), initial=0.0) > _REFINE_APPROACH_RAD
        )

    def _openness(self, joint_q: np.ndarray) -> np.ndarray:
        leaders = np.asarray(joint_q)[self.gripper_leaders]
        return np.clip((leaders - self.closed[0]) / self.gripper_span, 0.0, 1.0).astype(np.float32)

    def _replay(
        self,
        initial_q: np.ndarray,
        initial_qd: np.ndarray,
        arm_target: np.ndarray,
        gripper_command: np.ndarray,
        *,
        deterministic: bool = True,
    ) -> dict[str, np.ndarray]:
        runtime = self.replay_runtime if deterministic else self.runtime
        runtime.restore_rigid_state(initial_q, initial_qd)
        states = runtime.rigid_state()
        replay = {name: [value] for name, value in states.items()}
        for arm, gripper in zip(arm_target, gripper_command, strict=True):
            self._arm_target.assign(arm[None])
            self._gripper_command.assign(gripper[None])
            runtime.step_rigid(self._arm_target, self._gripper_command)
            state = runtime.rigid_state()
            for name in replay:
                replay[name].append(state[name])
        return {name: np.stack(value) for name, value in replay.items()}


def _one_step_acceleration(
    current_position: np.ndarray,
    current_velocity: np.ndarray,
    target_position: np.ndarray,
    target_velocity: np.ndarray,
    *,
    dt: float,
) -> np.ndarray:
    position = (target_position - current_position - dt * current_velocity) / (dt * dt)
    velocity = (target_velocity - current_velocity) / dt
    return 0.5 * (position + velocity)


def _inverse_position_drive_target(
    current_position: np.ndarray,
    current_velocity: np.ndarray,
    desired_position: np.ndarray,
    desired_velocity: np.ndarray,
    feedforward_effort: np.ndarray,
    *,
    stiffness: np.ndarray,
    damping: np.ndarray,
    effort_limit: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Invert a saturated position drive at one control boundary."""

    required = (
        np.asarray(feedforward_effort, dtype=np.float64)
        + stiffness * (desired_position - current_position)
        + damping * (desired_velocity - current_velocity)
    )
    applied = np.clip(required, -effort_limit, effort_limit)
    target = current_position + (applied + damping * current_velocity) / stiffness
    return np.clip(target, lower, upper), required


def _scale_effort_limits(model, dof_indices: np.ndarray, scale: float) -> None:
    """Scale selected actuator effort limits in every independent physics world."""

    factor = float(scale)
    if factor == 1.0:
        return
    world_count = int(model.world_count)
    dof_count = int(model.joint_dof_count) // world_count
    indices = np.asarray(dof_indices, dtype=np.int64)
    if np.any(indices < 0) or np.any(indices >= dof_count):
        raise RuntimeError("actuator indices differ from the model degree-of-freedom layout")
    limits = np.asarray(model.joint_effort_limit.numpy(), dtype=np.float32).reshape(
        world_count, dof_count
    )
    limits = limits.copy()
    limits[:, indices] *= factor
    model.joint_effort_limit.assign(limits.reshape(-1))


def _canonicalize_saturated_target(
    current_position: np.ndarray,
    current_velocity: np.ndarray,
    next_position: np.ndarray,
    next_velocity: np.ndarray,
    joint_target: np.ndarray,
    required_effort: np.ndarray,
    *,
    stiffness: np.ndarray,
    damping: np.ndarray,
    effort_limit: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Return the nearest target preserving saturation for a whole frame.

    A position drive is held over every physics substep.  A target that only
    reaches the effort limit at the control boundary can therefore become
    unsaturated partway through the frame.  The cubic Hermite boundary motion
    gives the smallest target that preserves the requested effort direction
    throughout the complete frame.
    """

    position = np.asarray(current_position, dtype=np.float64)
    velocity = np.asarray(current_velocity, dtype=np.float64)
    next_position = np.asarray(next_position, dtype=np.float64)
    next_velocity = np.asarray(next_velocity, dtype=np.float64)
    target = np.asarray(joint_target, dtype=np.float64).copy()
    required = np.asarray(required_effort, dtype=np.float64)
    positive = required > effort_limit
    negative = required < -effort_limit
    if not np.any(positive | negative):
        return target

    a0 = position
    a1 = dt * velocity
    a2 = -3.0 * position - 2.0 * a1 + 3.0 * next_position - dt * next_velocity
    a3 = 2.0 * position + a1 - 2.0 * next_position + dt * next_velocity
    damping_time = damping / (stiffness * dt)
    b0 = a0 + damping_time * a1
    b1 = a1 + 2.0 * damping_time * a2
    b2 = a2 + 3.0 * damping_time * a3
    b3 = a3
    interval_min = np.empty_like(position)
    interval_max = np.empty_like(position)
    for joint in range(len(position)):
        candidates = [0.0, 1.0]
        roots = np.roots(np.asarray((3.0 * b3[joint], 2.0 * b2[joint], b1[joint])))
        candidates.extend(
            float(root.real)
            for root in roots
            if abs(float(root.imag)) <= 1.0e-12 and 0.0 < float(root.real) < 1.0
        )
        u = np.asarray(candidates, dtype=np.float64)
        values = b0[joint] + u * (b1[joint] + u * (b2[joint] + u * b3[joint]))
        interval_min[joint] = float(np.min(values))
        interval_max[joint] = float(np.max(values))

    target[positive] = interval_max[positive] + (effort_limit[positive] / stiffness[positive])
    target[negative] = interval_min[negative] - (effort_limit[negative] / stiffness[negative])
    return np.clip(target, lower, upper)


def _attach_positions(body_q: np.ndarray, attach_frames, body_indices) -> np.ndarray:
    poses = np.asarray(body_q, dtype=np.float64)[:, list(body_indices)]
    offsets = np.asarray([frame.position for frame in attach_frames], dtype=np.float64)
    return poses[..., :3] + _quat_rotate(poses[..., 3:7], offsets)


def _attach_approach_directions(body_q: np.ndarray, attach_frames, body_indices) -> np.ndarray:
    poses = np.asarray(body_q, dtype=np.float64)[:, list(body_indices)]
    local_rotations = np.asarray([frame.rotation_xyzw for frame in attach_frames], dtype=np.float64)
    attach_rotations = _quat_multiply(poses[..., 3:7], local_rotations)
    return _quat_rotate(attach_rotations, np.asarray((0.0, 0.0, 1.0)))


def _task_space_residual(
    attach_positions: np.ndarray,
    openness: np.ndarray,
    approach_directions: np.ndarray,
    desired_attach_positions: np.ndarray,
    desired_openness: np.ndarray,
    desired_approach_directions: np.ndarray,
    pregrasp_mask: np.ndarray,
) -> np.ndarray:
    positions = np.asarray(attach_positions, dtype=np.float64)
    target_positions = np.asarray(desired_attach_positions, dtype=np.float64)
    open_state = np.asarray(openness, dtype=np.float64)
    target_open = np.asarray(desired_openness, dtype=np.float64)
    approach = _normalize_directions(approach_directions)
    target_approach = _normalize_directions(desired_approach_directions)
    active = np.asarray(pregrasp_mask, dtype=bool)
    if (
        positions.shape != (2, 3)
        or target_positions.shape != (2, 3)
        or open_state.shape != (2,)
        or target_open.shape != (2,)
        or approach.shape != (2, 3)
        or target_approach.shape != (2, 3)
        or active.shape != (2,)
    ):
        raise ValueError("task-space retarget values have invalid shapes")
    position_residual = (positions - target_positions).ravel() / _ATTACH_POSITION_LIMIT_M
    approach_residual = (
        (approach - target_approach) * active[:, None]
    ).ravel() / PREGRASP_APPROACH_TOLERANCE_RAD
    gripper_residual = (open_state - target_open) / ATTACH_OPENNESS_THRESHOLD
    return np.concatenate((position_residual, approach_residual, gripper_residual)).astype(
        np.float64
    )


def _normalize_directions(directions: np.ndarray) -> np.ndarray:
    values = np.asarray(directions, dtype=np.float64)
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    if not np.isfinite(values).all() or np.any(norm <= 1.0e-12):
        raise ValueError("approach directions must be finite and nonzero")
    return values / norm


def _direction_angle(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    lhs = _normalize_directions(current)
    rhs = _normalize_directions(target)
    cross = np.linalg.norm(np.cross(lhs, rhs), axis=-1)
    dot = np.clip(np.sum(lhs * rhs, axis=-1), -1.0, 1.0)
    return np.arctan2(cross, dot)


def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lhs = np.asarray(left, dtype=np.float64)
    rhs = np.broadcast_to(np.asarray(right, dtype=np.float64), lhs.shape)
    xyz = (
        lhs[..., 3:4] * rhs[..., :3]
        + rhs[..., 3:4] * lhs[..., :3]
        + np.cross(lhs[..., :3], rhs[..., :3])
    )
    scalar = lhs[..., 3:4] * rhs[..., 3:4] - np.sum(
        lhs[..., :3] * rhs[..., :3], axis=-1, keepdims=True
    )
    result = np.concatenate((xyz, scalar), axis=-1)
    return result / np.linalg.norm(result, axis=-1, keepdims=True)


def _quat_rotate(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    v = np.broadcast_to(np.asarray(vector, dtype=np.float64), q[..., :3].shape)
    xyz = q[..., :3]
    return v + 2.0 * np.cross(xyz, np.cross(xyz, v) + q[..., 3:4] * v)


def _max(values: np.ndarray) -> float:
    return float(np.max(np.asarray(values), initial=0.0))


def _quantile(values: np.ndarray, fraction: float) -> float:
    array = np.asarray(values)
    return float(np.quantile(array, fraction)) if array.size else 0.0


def _select_repair_window(
    attach_errors: np.ndarray,
    approach_errors: np.ndarray,
    pregrasp_mask: np.ndarray,
    *,
    action_count: int,
    horizon: int,
    attempted: set[tuple[int, int]],
) -> tuple[int, int, int] | None:
    errors = np.asarray(attach_errors, dtype=np.float64)
    direction_errors = np.asarray(approach_errors, dtype=np.float64)
    approach_mask = np.asarray(pregrasp_mask, dtype=bool)
    normalized = np.maximum.reduce(
        (
            errors / _ATTACH_POSITION_LIMIT_M,
            np.where(
                approach_mask,
                direction_errors / PREGRASP_APPROACH_TOLERANCE_RAD,
                0.0,
            ),
        )
    )
    scores = np.max(normalized, axis=1)
    scores[0] = -np.inf
    for failing_state in np.argsort(scores)[::-1]:
        if scores[failing_state] <= 1.0:
            return None
        has_recovery_step = failing_state < action_count and horizon > 1
        start = int(failing_state) - horizon + int(has_recovery_step)
        start = min(max(start, 0), action_count - horizon)
        end = start + horizon
        if (start, end) not in attempted:
            return start, end, int(failing_state)
    return None


def _window_candidate_improves(
    baseline: tuple[float, float],
    candidate: tuple[float, float],
) -> bool:
    tolerance = 1.0e-6
    baseline_score = max(
        baseline[0] / _ATTACH_POSITION_LIMIT_M,
        baseline[1] / PREGRASP_APPROACH_TOLERANCE_RAD,
    )
    candidate_score = max(
        candidate[0] / _ATTACH_POSITION_LIMIT_M,
        candidate[1] / PREGRASP_APPROACH_TOLERANCE_RAD,
    )
    position_ceiling = (
        _ATTACH_POSITION_LIMIT_M
        if baseline[0] <= _ATTACH_POSITION_LIMIT_M
        else baseline[0] + tolerance
    )
    approach_ceiling = (
        PREGRASP_APPROACH_TOLERANCE_RAD
        if baseline[1] <= PREGRASP_APPROACH_TOLERANCE_RAD
        else baseline[1] + tolerance
    )
    return bool(
        candidate_score < baseline_score - tolerance
        and candidate[0] <= position_ceiling
        and candidate[1] <= approach_ceiling
    )

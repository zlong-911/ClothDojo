"""Dynamics-consistent target-robot actions using the workstation's USD drives."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import least_squares

from clothdojo.robots import ALOHA2, robot_profile
from clothdojo.sim.grasp import (
    _GRIPPER_JOINT_MARGIN,
    ATTACH_OPENNESS_THRESHOLD,
    RELEASE_OPENNESS_THRESHOLD,
)
from clothdojo.sim.runtime import ShirtFlattenRuntime
from clothdojo.sim.scene import ShirtFlattenSpec, build_scene

_BASE_MAX_NFEV = 3
_REFINED_MAX_NFEV = 12
_REFINE_ATTACH_POSITION_M = 0.003
_REFINE_APPROACH_RAD = 0.02
_REFINE_GRIPPER_OPENNESS = 0.01
_ARM_POSITION_LIMIT_RAD = 0.025
_ARM_VELOCITY_LIMIT_RAD_S = 1.0
_ATTACH_POSITION_LIMIT_M = 0.01
PREGRASP_APPROACH_TOLERANCE_RAD = 0.05
_ACTION_SEED_RESIDUAL_SCALE = 0.03
_WINDOW_TERMINAL_POSITION_SCALE = 0.1
_WINDOW_TERMINAL_VELOCITY_SCALE = 0.01
_ARM_TASK_RESIDUAL_SIZE = 18
_TASK_RESIDUAL_SIZE = _ARM_TASK_RESIDUAL_SIZE + 2
_ACTION_REPLAY_POSITION_LIMIT_RAD = 1.0e-4
_ACTION_REPLAY_VELOCITY_LIMIT_RAD_S = 1.0e-3
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


class RobotDynamicsRetargeter:
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
        def build_runtime(world_count: int) -> ShirtFlattenRuntime:
            spec = ShirtFlattenSpec(
                garment_id=garment_id,
                world_count=world_count,
                control_fps=int(rounded_fps),
                sim_substeps=int(sim_substeps),
            )
            scene = build_scene(
                Path(assets_root), spec=spec, device=device, robot_id=target_robot,
                visuals=False, include_cloth=False,
                robot_cloth_collision=False, robot_rigid_collision=False,
            )
            return ShirtFlattenRuntime(
                scene, use_cuda_graph=False, rigid_deterministic=False,
                rigid_cuda_graph=sim_substeps % 2 == 0,
            )

        # The three engines share configuration, not mutable model/solver state.
        self.runtime = build_runtime(1)
        self.replay_runtime = build_runtime(1)
        self.shooting_runtime = build_runtime(self.arm_count + 3)
        self.shooting_world_count = int(self.shooting_runtime.model.world_count)
        self.shooting_world_offsets = np.asarray(
            self.shooting_runtime.scene.world_offsets, dtype=np.float32,
        )
        self.dt = self.runtime.control_dt
        self.gripper = np.asarray(self.profile.gripper_coord_indices, dtype=np.int64)
        self.gripper_dof = profile_gripper_dof
        gripper_stride = self.profile.gripper_coords_per_side
        self.gripper_leaders = self.gripper[[0, gripper_stride]]
        self.gripper_dof_leaders = self.gripper_dof[[0, gripper_stride]]
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
        *,
        target_up_directions: np.ndarray | None = None,
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
        if np.any(pregrasp) or np.any(approach_validation):
            raise ValueError(
                "position-only dynamics retarget does not accept orientation constraints"
            )
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
        target_up = None
        if target_up_directions is not None:
            target_up = np.asarray(target_up_directions, dtype=np.float32)
            if (
                target_up.shape != target_approach.shape
                or not np.isfinite(target_up).all()
                or np.any(np.abs(np.linalg.norm(target_up, axis=-1) - 1) > 1e-4)
                or np.any(np.abs(np.sum(target_up * target_approach, axis=-1)) > 1e-4)
            ):
                raise ValueError("target up directions must be unit axes perpendicular to approach")
            # Direction references are diagnostic only; roll preference lives
            # in the geometric seed, never in the dynamics objective or gate.

        self.runtime.restore_rigid_state(reference_q[0], reference_qd[0])
        # Feedback needs only the current joint boundary. Full body histories
        # are produced later by the independent action-only replay.
        current = self.runtime.rigid_state(("joint_q", "joint_qd"))
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
            current_q = current["joint_q"]
            current_qd = current["joint_qd"]
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
            inverse_target, required_effort = _inverse_position_drive_target(
                current_q[self.arm],
                current_qd[self.arm_dof],
                reference_q[index, self.arm],
                reference_qd[index, self.arm_dof],
                feedforward,
                stiffness=self.stiffness,
                damping=self.damping,
                effort_limit=self.effort_limit,
                lower=self.lower,
                upper=self.upper,
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
                reference_q[index, self.gripper_leaders],
                reference_qd[index, self.gripper_dof_leaders],
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
            # Openness maps to the profile's finger position target. Like arm
            # commands, this inverse-drive seed is refined by forward shooting.
            # Event preview later advances the command, not the state timeline.
            desired_open = reference_open[index + 1]
            desired_attach = target_attach[index + 1]
            desired_approach = target_approach[index + 1]
            desired_pregrasp = pregrasp[index + 1]
            desired_up = None if target_up is None else target_up[index + 1]
            arm_candidates = [inverse_target.copy()]
            for free_target in (
                previous_command,
                reference_q[index + 1, self.arm],
                propagated,
            ):
                arm_candidates.append(np.asarray(free_target, dtype=np.float64).copy())
            # Very stiff USD drives have broad flat action regions after effort
            # clipping. Keep shooting inside the useful target interval along
            # the predicted boundary motion; otherwise finite differences can
            # be identically zero while the optimizer drifts to a joint limit.
            arm_lower, arm_upper = _position_drive_search_bounds(
                current_q[self.arm], current_qd[self.arm_dof],
                reference_q[index + 1, self.arm], reference_qd[index + 1, self.arm_dof],
                stiffness=self.stiffness, damping=self.damping,
                effort_limit=self.effort_limit, lower=self.lower, upper=self.upper, dt=self.dt,
            )
            candidates = np.stack(
                [np.concatenate((np.clip(candidate, arm_lower, arm_upper), grip))
                 for candidate in arm_candidates]
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
                desired_up,
            )
            trial_evaluations += len(candidates)
            initial_slot = int(
                np.argmin(np.sum(candidate_residuals[:, :_ARM_TASK_RESIDUAL_SIZE] ** 2, axis=1))
            )
            initial_command = candidates[initial_slot].copy()
            initial_residual = candidate_residuals[initial_slot]

            def task_residual(
                values: np.ndarray,
                state=branch_state,
                target_open=desired_open,
                target_attach_position=desired_attach,
                target_approach_direction=desired_approach,
                target_pregrasp=desired_pregrasp,
                target_up_direction=desired_up,
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
                    target_up_direction,
                )

            command = initial_command.copy()
            best_residual = np.concatenate(
                (initial_residual[:_ARM_TASK_RESIDUAL_SIZE], np.zeros(self.arm_count))
            )
            action_lower = np.concatenate((arm_lower, np.zeros(2)))
            action_upper = np.concatenate((arm_upper, np.ones(2)))
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
                            task_residual(action)[:_ARM_TASK_RESIDUAL_SIZE],
                            _ACTION_SEED_RESIDUAL_SCALE
                            * (action[: self.arm_count] - seed[: self.arm_count]),
                        )
                    )

                def free_jacobian(
                    values: np.ndarray,
                    action: np.ndarray = command,
                    free_indices: np.ndarray = free_action,
                    lower: np.ndarray = action_lower,
                    upper: np.ndarray = action_upper,
                    state: tuple[np.ndarray, np.ndarray] = branch_state,
                    seed: np.ndarray = initial_command,
                    target_open: np.ndarray = desired_open,
                    target_attach_position: np.ndarray = desired_attach,
                    target_approach_direction: np.ndarray = desired_approach,
                    target_pregrasp: np.ndarray = desired_pregrasp,
                    target_up_direction: np.ndarray | None = desired_up,
                ) -> np.ndarray:
                    nonlocal batched_jacobian_steps
                    nonlocal batched_jacobian_world_evaluations
                    base_action = action.copy()
                    base_action[free_indices] = values
                    commands = np.tile(base_action, (len(free_indices) + 1, 1))
                    steps = 1.0e-3 * np.maximum(1.0, np.abs(values))
                    forward = upper[free_indices] - values
                    backward = values - lower[free_indices]
                    positive = forward >= np.minimum(steps, backward)
                    steps = np.where(
                        positive, np.minimum(steps, forward), -np.minimum(steps, backward)
                    )
                    commands[1:, free_indices] += np.diag(steps)
                    task_residuals = self._batch_trial_residuals(
                        state,
                        commands,
                        target_open,
                        target_attach_position,
                        target_approach_direction,
                        target_pregrasp,
                        target_up_direction,
                    )
                    residuals = np.concatenate(
                        (
                            task_residuals[:, :_ARM_TASK_RESIDUAL_SIZE],
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

            # Solve the gripper position commands with the same bounded forward
            # shooting, but hold the selected arm command fixed. Otherwise an
            # unreachable finger target can make the optimizer swing the arm
            # to exploit inertial forces instead of tracking the arm task.
            gripper_seed = command[self.arm_count :].copy()

            def gripper_residual(
                values: np.ndarray,
                action: np.ndarray = command,
                seed: np.ndarray = gripper_seed,
            ) -> np.ndarray:
                trial_command = action.copy()
                trial_command[self.arm_count :] = values
                return np.concatenate((
                    task_residual(trial_command)[_ARM_TASK_RESIDUAL_SIZE:],
                    _ACTION_SEED_RESIDUAL_SCALE * (values - seed),
                ))

            def gripper_jacobian(
                values: np.ndarray,
                action: np.ndarray = command,
                state: tuple[np.ndarray, np.ndarray] = branch_state,
                target_open: np.ndarray = desired_open,
                target_attach_position: np.ndarray = desired_attach,
                target_approach_direction: np.ndarray = desired_approach,
                target_pregrasp: np.ndarray = desired_pregrasp,
                target_up_direction: np.ndarray | None = desired_up,
            ) -> np.ndarray:
                nonlocal batched_jacobian_steps
                nonlocal batched_jacobian_world_evaluations
                commands = np.tile(action, (3, 1))
                commands[:, self.arm_count :] = values
                steps = np.where(values + 1.0e-3 <= 1.0, 1.0e-3, -1.0e-3)
                commands[1:, self.arm_count :] += np.diag(steps)
                residuals = self._batch_trial_residuals(
                    state, commands, target_open, target_attach_position,
                    target_approach_direction, target_pregrasp,
                    target_up_direction,
                )[:, _ARM_TASK_RESIDUAL_SIZE:]
                batched_jacobian_steps += 1
                batched_jacobian_world_evaluations += len(commands)
                return np.concatenate((
                    ((residuals[1:] - residuals[0]) / steps[:, None]).T,
                    _ACTION_SEED_RESIDUAL_SCALE * np.eye(2),
                ))

            initial_gripper_residual = gripper_residual(gripper_seed)
            if self._needs_gripper_refinement(initial_gripper_residual):
                fit = least_squares(
                    gripper_residual, gripper_seed, bounds=(np.zeros(2), np.ones(2)),
                    jac=gripper_jacobian, max_nfev=_REFINED_MAX_NFEV,
                    x_scale="jac", xtol=1.0e-7, ftol=1.0e-7, gtol=1.0e-7,
                )
                optimizer_evaluations += int(fit.nfev)
                if np.dot(fit.fun, fit.fun) < np.dot(
                    initial_gripper_residual, initial_gripper_residual
                ):
                    command[self.arm_count :] = fit.x
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
            current = self.runtime.rigid_state(("joint_q", "joint_qd"))
            arm_targets.append(command[: self.arm_count].astype(np.float32))
            gripper_commands.append(command[self.arm_count :].astype(np.float32))

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
            target_up,
        )
        window_refinement_seconds = perf_counter() - window_started
        # Finish arm refinement before aligning gripper events on those actions.
        preview_started = perf_counter()
        action_gripper, arrays, preview_stats = self._align_gripper_events(
            action_q, action_gripper, reference_open, attached_mask, arrays
        )
        preview_seconds = perf_counter() - preview_started
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
        finger_replay_position_error = np.abs(
            self._finger_openness(replay["joint_q"])
            - self._finger_openness(arrays["joint_q"])
        )
        finger_replay_velocity_error = np.abs(
            (replay["joint_qd"][:, self.gripper_dof] - arrays["joint_qd"][:, self.gripper_dof])
            / np.tile(self.opened - self.closed, 2)
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
        finger_open = self._finger_openness(arrays["joint_q"])
        event_validation = _gripper_event_validation(finger_open, attached_mask)
        event_open = _gripper_event_openness(finger_open)
        open_array = np.clip(finger_open.mean(axis=-1), 0.0, 1.0).astype(np.float32)
        open_error = np.max(np.abs(finger_open - reference_open[..., None]), axis=-1)
        sync_error = np.ptp(finger_open, axis=-1)
        attached_open_error = open_error[attached_mask]
        attach_rising = np.zeros_like(attached_mask)
        attach_rising[0] = attached_mask[0]
        attach_rising[1:] = attached_mask[1:] & ~attached_mask[:-1]
        attach_transition_open = event_open[attach_rising]
        release_falling = np.zeros_like(attached_mask)
        release_falling[1:] = ~attached_mask[1:] & attached_mask[:-1]
        release_transition_open = event_open[release_falling]
        approach_directions = _attach_approach_directions(
            arrays["body_q"], self.profile.attach_frames, self.attach_body_indices
        )
        approach_errors = _direction_angle(approach_directions, target_approach)
        pregrasp_approach_errors = approach_errors[approach_validation]
        up_errors = np.zeros_like(approach_errors)
        if target_up is not None:
            up_errors = _direction_angle(_attach_axis_directions(
                arrays["body_q"], self.profile.attach_frames, self.attach_body_indices,
                (0., 1., 0.),
            ), target_up)
        metrics = {
            "source_timeline_preserved": True,
            "robot_dynamics_rollout": True,
            "style3d_solver_used": False,
            "actuator_effort_scale": 1.0,
            "gripper_event_validation": event_validation,
            "gripper_event_preview": preview_stats,
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
            "max_gripper_finger_sync_error": _max(sync_error),
            "max_attached_gripper_openness_error": _max(attached_open_error),
            "max_attach_transition_gripper_openness": _max(attach_transition_open),
            "min_release_transition_gripper_openness": float(
                np.min(release_transition_open, initial=1.0)
            ),
            "max_attach_position_error_m": _max(attach_errors),
            "max_attached_position_error_m": _max(active_errors),
            "p95_attached_position_error_m": _quantile(active_errors, 0.95),
            "max_pregrasp_approach_error_rad": _max(pregrasp_approach_errors),
            "up_axis_constrained": False,
            "orientation_constraint_scope": "none",
            "max_up_axis_error_rad": _max(up_errors[approach_validation]),
            "p95_pregrasp_approach_error_rad": _quantile(pregrasp_approach_errors, 0.95),
            "effort_saturation_fraction": (effort_saturation_count / action_value_count),
            "maximum_required_effort_ratio": maximum_required_effort_ratio,
            "gripper_effort_saturation_fraction": (
                gripper_effort_saturation_count / gripper_action_value_count
            ),
            "maximum_required_gripper_effort_ratio": maximum_required_gripper_effort_ratio,
            "action_replay_position_error_rad": _max(replay_position_error),
            "action_replay_velocity_error_rad_s": _max(replay_velocity_error),
            "action_replay_gripper_openness_error": _max(finger_replay_position_error),
            "action_replay_gripper_openness_velocity_error_s": _max(finger_replay_velocity_error),
            "limits": {
                "attach_transition_gripper_openness": ATTACH_OPENNESS_THRESHOLD,
                "release_transition_gripper_openness": RELEASE_OPENNESS_THRESHOLD,
                "gripper_event_frame_error": 0,
                "attach_position_error_m": _ATTACH_POSITION_LIMIT_M,
                "action_replay_position_error_rad": _ACTION_REPLAY_POSITION_LIMIT_RAD,
                "action_replay_velocity_error_rad_s": (_ACTION_REPLAY_VELOCITY_LIMIT_RAD_S),
                "action_replay_gripper_openness_error": _ACTION_REPLAY_POSITION_LIMIT_RAD,
                "action_replay_gripper_openness_velocity_error_s": (
                    _ACTION_REPLAY_VELOCITY_LIMIT_RAD_S
                ),
            },
            "diagnostic_thresholds": {
                "ik_seed_position_deviation_rad": _ARM_POSITION_LIMIT_RAD,
                "ik_seed_velocity_deviation_rad_s": _ARM_VELOCITY_LIMIT_RAD_S,
            },
            "timing_seconds": {
                "inverse_dynamics": inverse_dynamics_seconds,
                "shooting": shooting_seconds,
                "gripper_event_preview": preview_seconds,
                "window_refinement": window_refinement_seconds,
                "validation_replays": validation_replay_seconds,
                "total": perf_counter() - total_started,
            },
        }
        metrics["dynamics_valid"] = bool(
            event_validation["valid"]
            and metrics["max_attach_position_error_m"] <= _ATTACH_POSITION_LIMIT_M
            and metrics["action_replay_position_error_rad"] <= _ACTION_REPLAY_POSITION_LIMIT_RAD
            and metrics["action_replay_velocity_error_rad_s"] <= _ACTION_REPLAY_VELOCITY_LIMIT_RAD_S
            and metrics["action_replay_gripper_openness_error"] <= _ACTION_REPLAY_POSITION_LIMIT_RAD
            and metrics["action_replay_gripper_openness_velocity_error_s"]
            <= _ACTION_REPLAY_VELOCITY_LIMIT_RAD_S
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
        return self.runtime.rigid_state(("joint_q", "body_q"))

    def _align_gripper_events(
        self,
        arm_commands: np.ndarray,
        gripper_commands: np.ndarray,
        reference_open: np.ndarray,
        attached: np.ndarray,
        arrays: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, dict[str, np.ndarray], dict]:
        """Preview each event and shift only its bounded gripper command curve.

        Arm commands, state sample times and the source event frames stay fixed.
        Each trial starts from an actual rigid boundary, not a reference pose.
        Fractional-frame command interpolation changes the feedforward command,
        never the output state/cloth timeline.
        """
        commands = gripper_commands.copy()
        prefix = {name: value.copy() for name, value in arrays.items()}
        prefix_end = 0
        phase_start = np.zeros(2, dtype=np.int64)
        events = np.argwhere(attached[1:] != attached[:-1])
        stats = {"search_batches": 0, "preview_world_steps": 0, "events": []}

        def advance(start: int, end: int) -> None:
            segment = self._replay(
                prefix["joint_q"][start], prefix["joint_qd"][start],
                arm_commands[start:end], commands[start:end],
            )
            for name in prefix:
                prefix[name][start : end + 1] = segment[name]

        for transition, side in events:
            frame, side = int(transition) + 1, int(side)
            closing = bool(attached[frame, side])
            direction = -1.0 if closing else 1.0
            motion_start = frame
            while motion_start > phase_start[side] and direction * (
                reference_open[motion_start, side] - reference_open[motion_start - 1, side]
            ) > 1.0e-4:
                motion_start -= 1
            # Start with one extra source transition duration as preview. If
            # that is insufficient, grow only within the preceding phase.
            start = max(int(phase_start[side]), frame - 2 * max(1, frame - motion_start))
            best_margin, best_lead = -np.inf, 0.0
            while True:
                if start > prefix_end:
                    advance(prefix_end, start)
                    prefix_end = start
                state = (prefix["joint_q"][start], prefix["joint_qd"][start])
                base = commands[start:frame].copy()
                upper = float(frame - start - 1)
                lower = -upper
                best_margin, best_lead = -np.inf, 0.0
                for refinement in range(5):
                    leads = np.linspace(lower, upper, self.shooting_world_count)
                    candidates = np.repeat(base[None], len(leads), axis=0)
                    for slot, lead in enumerate(leads):
                        candidates[slot, :, side] = _advance_gripper_curve(base[:, side], lead)
                    fingers = self._preview_gripper_commands(
                        state, arm_commands[start:frame], candidates
                    )[:, :, side]
                    margins = _gripper_event_margin(fingers, closing=closing)
                    stats["search_batches"] += 1
                    stats["preview_world_steps"] += len(leads) * (frame - start)
                    best = int(np.argmax(margins))
                    if margins[best] > best_margin:
                        best_margin, best_lead = float(margins[best]), float(leads[best])
                    lower = float(leads[max(0, best - 1)])
                    upper = float(leads[min(len(leads) - 1, best + 1)])
                    if upper == lower or (refinement >= 2 and best_margin > 1.0e-4):
                        break
                at_preview_limit = _gripper_preview_at_limit(best_lead, frame - start)
                if best_margin > 0.0 or not at_preview_limit or start == phase_start[side]:
                    break
                start = max(int(phase_start[side]), frame - 2 * (frame - start))

            commands[start:frame, side] = _advance_gripper_curve(base[:, side], best_lead)
            advance(start, frame)
            prefix_end = frame
            phase_start[side] = frame
            stats["events"].append({
                "side": side, "frame": frame, "event": "attach" if closing else "release",
                "command_lead_frames": best_lead, "threshold_margin": best_margin,
            })

        # A single cold replay, including the unmodified suffix, is the source
        # of truth. It also checks effects on the opposite hand and arm motion.
        replay = self._replay(
            arrays["joint_q"][0], arrays["joint_qd"][0], arm_commands, commands
        )
        return commands, replay, stats

    def _preview_gripper_commands(
        self,
        state: tuple[np.ndarray, np.ndarray],
        arm_commands: np.ndarray,
        gripper_commands: np.ndarray,
    ) -> np.ndarray:
        count, horizon, _ = gripper_commands.shape
        if count != self.shooting_world_count or len(arm_commands) != horizon:
            raise ValueError("gripper preview differs from the shooting world layout")
        self.shooting_runtime.restore_rigid_state_batch(
            np.repeat(state[0][None], count, axis=0),
            np.repeat(state[1][None], count, axis=0),
        )
        predicted = self.shooting_runtime.rollout_rigid(
            np.repeat(arm_commands[:, None], count, axis=1),
            gripper_commands.transpose(1, 0, 2), fields=("joint_q",),
        )
        return self._finger_openness(predicted["joint_q"]).swapaxes(0, 1)

    def _batch_trial_residuals(
        self,
        state: tuple[np.ndarray, np.ndarray],
        commands: np.ndarray,
        desired_open: np.ndarray,
        desired_attach_positions: np.ndarray,
        desired_approach_directions: np.ndarray,
        pregrasp_mask: np.ndarray,
        desired_up_directions: np.ndarray | None = None,
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
        predicted = self.shooting_runtime.rigid_state_batch(("joint_q", "body_q"))
        body_q = predicted["body_q"][:count].copy()
        body_q[..., :3] -= self.shooting_world_offsets[:count, None, :]
        return self._residual_batch(
            predicted["joint_q"][:count], body_q, desired_open,
            desired_attach_positions, desired_approach_directions,
            pregrasp_mask, desired_up_directions,
        )

    def _batch_window_residuals(
        self,
        state: tuple[np.ndarray, np.ndarray],
        command_sequences: np.ndarray,
        desired_open: np.ndarray,
        desired_attach_positions: np.ndarray,
        desired_approach_directions: np.ndarray,
        pregrasp_mask: np.ndarray,
        desired_up_directions: np.ndarray | None = None,
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
            predicted = self.shooting_runtime.rigid_state_batch(("joint_q", "joint_qd", "body_q"))
            body_q = predicted["body_q"][:count].copy()
            body_q[..., :3] -= self.shooting_world_offsets[:count, None, :]
            step_residuals.append(
                self._residual_batch(
                    predicted["joint_q"][:count], body_q, desired_open[step],
                    desired_attach_positions[step], desired_approach_directions[step],
                    pregrasp_mask[step],
                    None if desired_up_directions is None else desired_up_directions[step],
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
        target_up: np.ndarray | None = None,
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
            up_directions = _attach_axis_directions(
                arrays["body_q"], self.profile.attach_frames, self.attach_body_indices,
                (0., 1., 0.),
            )
            if target_up is not None:
                approach_errors = np.maximum(
                    approach_errors, _direction_angle(up_directions, target_up)
                )
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
            desired_up = None if target_up is None else target_up[start + 1 : end + 1].copy()
            if end > failing_state:
                desired_attach[-1] = attach_positions[end]
                desired_approach[-1] = approach_directions[end]
                if desired_up is not None:
                    desired_up[-1] = up_directions[end]

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
                tracked = shaped[:, steps, :_ARM_TASK_RESIDUAL_SIZE].reshape(len(task_values), -1)
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
                target_up_directions: np.ndarray | None = desired_up,
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
                    target_up_directions,
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
                target_up_directions: np.ndarray | None = desired_up,
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
                            target_up_directions,
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
            )
            baseline_values = self._window_validation_values(
                arrays,
                target_attach,
                target_approach,
                approach_validation,
                target_up,
            )
            candidate_values = self._window_validation_values(
                candidate_arrays,
                target_attach,
                target_approach,
                approach_validation,
                target_up,
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
        target_up: np.ndarray | None = None,
    ) -> tuple[float, float]:
        positions = _attach_positions(
            arrays["body_q"], self.profile.attach_frames, self.attach_body_indices
        )
        position_errors = np.linalg.norm(positions - target_attach, axis=-1)
        approach = _attach_approach_directions(
            arrays["body_q"], self.profile.attach_frames, self.attach_body_indices
        )
        approach_errors = _direction_angle(approach, target_approach)[approach_validation]
        up_error = 0.0
        if target_up is not None:
            up = _attach_axis_directions(
                arrays["body_q"], self.profile.attach_frames, self.attach_body_indices,
                (0., 1., 0.),
            )
            up_error = _max(_direction_angle(up, target_up)[approach_validation])
        return _max(position_errors), max(_max(approach_errors), up_error)

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
        desired_up_directions: np.ndarray | None = None,
    ) -> np.ndarray:
        return self._residual_batch(
            np.asarray(joint_q)[None], np.asarray(body_q)[None], desired_open,
            desired_attach_positions, desired_approach_directions, pregrasp_mask,
            desired_up_directions,
        )[0]

    def _residual_batch(
        self,
        joint_q: np.ndarray,
        body_q: np.ndarray,
        desired_open: np.ndarray,
        desired_attach_positions: np.ndarray,
        desired_approach_directions: np.ndarray,
        pregrasp_mask: np.ndarray,
        desired_up_directions: np.ndarray | None = None,
    ) -> np.ndarray:
        """Evaluate shooting worlds together; inactive orientation costs are zero."""
        poses = np.asarray(body_q, dtype=np.float32)
        attach_position = _attach_positions(
            poses, self.profile.attach_frames, self.attach_body_indices,
        )
        attach_approach = desired_approach_directions
        attach_up = desired_up_directions
        if np.any(pregrasp_mask):
            attach_approach = _attach_approach_directions(
                poses, self.profile.attach_frames, self.attach_body_indices,
            )
            if desired_up_directions is not None:
                attach_up = _attach_axis_directions(
                    poses, self.profile.attach_frames, self.attach_body_indices, (0., 1., 0.),
                )
        return _task_space_residual(
            attach_position,
            self._openness(joint_q),
            attach_approach,
            desired_attach_positions,
            desired_open,
            desired_approach_directions,
            pregrasp_mask,
            up_directions=attach_up,
            desired_up_directions=desired_up_directions,
        )

    def _needs_arm_refinement(self, residual: np.ndarray) -> bool:
        values = np.asarray(residual, dtype=np.float64)[:_ARM_TASK_RESIDUAL_SIZE]
        attach = values[:6].reshape(2, 3) * _ATTACH_POSITION_LIMIT_M
        approach = values[6:12].reshape(2, 3) * PREGRASP_APPROACH_TOLERANCE_RAD
        up = values[12:18].reshape(2, 3) * PREGRASP_APPROACH_TOLERANCE_RAD
        return bool(
            np.max(np.linalg.norm(attach, axis=-1), initial=0.0) > _REFINE_ATTACH_POSITION_M
            or np.max(np.linalg.norm(approach, axis=-1), initial=0.0) > _REFINE_APPROACH_RAD
            or np.max(np.linalg.norm(up, axis=-1), initial=0.0) > _REFINE_APPROACH_RAD
        )

    def _needs_gripper_refinement(self, residual: np.ndarray) -> bool:
        return (
            _max(np.abs(np.asarray(residual)[:2])) * ATTACH_OPENNESS_THRESHOLD
            > _REFINE_GRIPPER_OPENNESS
        )

    def _finger_openness(self, joint_q: np.ndarray) -> np.ndarray:
        fingers = np.asarray(joint_q)[..., self.gripper].reshape(
            (*np.asarray(joint_q).shape[:-1], 2, self.profile.gripper_coords_per_side)
        )
        return (fingers - self.closed) / (self.opened - self.closed)

    def _openness(self, joint_q: np.ndarray) -> np.ndarray:
        # Keep the residual unclipped so an out-of-range finger is not hidden
        # behind a seemingly perfect fully-open/closed observation.
        return self._finger_openness(joint_q).mean(axis=-1)

    def _replay(
        self,
        initial_q: np.ndarray,
        initial_qd: np.ndarray,
        arm_target: np.ndarray,
        gripper_command: np.ndarray,
    ) -> dict[str, np.ndarray]:
        # Use the same numerical mode as shooting and normal runtime, with a
        # separate cold-started solver so replay does not consume branch state.
        runtime = self.replay_runtime
        runtime.restore_rigid_state(initial_q, initial_qd)
        return {
            name: values[:, 0]
            for name, values in runtime.rollout_rigid(
                arm_target[:, None], gripper_command[:, None],
            ).items()
        }


def _advance_gripper_curve(commands: np.ndarray, lead_frames: float) -> np.ndarray:
    """Shift a local curve; negative lead delays it, holding endpoint values."""
    values = np.asarray(commands, dtype=np.float64)
    frames = np.arange(len(values), dtype=np.float64)
    return np.interp(frames + float(lead_frames), frames, values)


def _gripper_preview_at_limit(lead_frames: float, window_frames: int) -> bool:
    """Return whether either shift direction exhausted the preview window."""

    if isinstance(window_frames, bool) or not isinstance(window_frames, (int, np.integer)):
        raise ValueError("gripper preview window must be an integer")
    if window_frames < 1:
        raise ValueError("gripper preview window must be positive")
    return bool(abs(float(lead_frames)) >= window_frames - 1 - 1.0e-6)


def _gripper_event_openness(finger_open: np.ndarray) -> np.ndarray:
    """ParticleGrasp's leader signal, from unadjusted per-finger openness.

    The runtime applies its soft endpoint margin to the signed joint range.
    Applying the same margin after normalization is algebraically equivalent,
    including grippers that open toward decreasing joint positions. Followers
    remain part of state/replay/synchronization diagnostics, not event triggers.
    """
    leader = np.asarray(finger_open, dtype=np.float64)[..., 0]
    return np.clip(
        (leader - _GRIPPER_JOINT_MARGIN) / (1.0 - 2.0 * _GRIPPER_JOINT_MARGIN),
        0.0, 1.0,
    )


def _gripper_event_margin(finger_open: np.ndarray, *, closing: bool) -> np.ndarray:
    """Positive only if the runtime leader first crosses in the final frame."""
    threshold = ATTACH_OPENNESS_THRESHOLD if closing else RELEASE_OPENNESS_THRESHOLD
    signed = (_gripper_event_openness(finger_open) - threshold) * (-1.0 if closing else 1.0)
    return np.minimum(-signed[:, :-1].max(axis=1), signed[:, -1])


def _gripper_event_validation(finger_open: np.ndarray, attached: np.ndarray) -> dict:
    """Recompute ParticleGrasp leader events at recorded control boundaries.

    These are opening-threshold events, not a cloth-contact simulation. The
    source attachment annotations specify the initial state and expected frame
    sequence only. Substep capture timing is not observable in these samples.
    """
    fingers = np.asarray(finger_open, dtype=np.float64)
    expected = np.asarray(attached, dtype=bool)
    if (fingers.ndim != 3 or fingers.shape[:2] != expected.shape
            or not len(fingers) or not fingers.shape[2]):
        raise ValueError("finger states and source attachment frames must have matching shapes")
    if not np.isfinite(fingers).all():
        raise ValueError("finger openness must be finite")
    openness = _gripper_event_openness(fingers)
    actual = np.empty(openness.shape, dtype=bool)
    actual[0] = expected[0]
    for frame in range(1, len(actual)):
        crossed = (openness[frame - 1] >= ATTACH_OPENNESS_THRESHOLD) & (
            openness[frame] < ATTACH_OPENNESS_THRESHOLD
        )
        actual[frame] = np.where(
            actual[frame - 1], openness[frame] <= RELEASE_OPENNESS_THRESHOLD, crossed
        )
    mismatches = actual != expected
    sides = []
    maximum_error = 0
    counts_match = True
    for side in range(expected.shape[1]):
        source_frames = np.flatnonzero(expected[1:, side] != expected[:-1, side]) + 1
        frames = np.flatnonzero(actual[1:, side] != actual[:-1, side]) + 1
        if len(frames) != len(source_frames):
            counts_match = False
        elif len(frames):
            maximum_error = max(maximum_error, int(np.max(np.abs(frames - source_frames))))
        sides.append({
            "side": side, "source_frames": source_frames.tolist(),
            "source_attached": expected[source_frames, side].tolist(),
            "runtime_frames": frames.tolist(),
        })
    return {
        "valid": not bool(mismatches.any()),
        "openness_source": "runtime_leader_with_joint_margin",
        "joint_margin": _GRIPPER_JOINT_MARGIN,
        "sampling": "control_frame",
        "mismatched_frames": int(np.count_nonzero(mismatches)),
        "event_counts_match": counts_match,
        "max_frame_error": maximum_error if counts_match else None,
        "sides": sides,
    }


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


def _position_drive_search_bounds(
    current_position: np.ndarray,
    current_velocity: np.ndarray,
    next_position: np.ndarray,
    next_velocity: np.ndarray,
    *,
    stiffness: np.ndarray,
    damping: np.ndarray,
    effort_limit: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Bound numerical search by the two full-frame saturated drive targets."""
    bounds = [
        _canonicalize_saturated_target(
            current_position, current_velocity, next_position, next_velocity,
            current_position, sign * 2 * effort_limit,
            stiffness=stiffness, damping=damping, effort_limit=effort_limit,
            lower=lower, upper=upper, dt=dt,
        )
        for sign in (-1.0, 1.0)
    ]
    # At a physical limit even the entire useful interval may be out of range.
    # Retain a tiny legal interval for scipy's strict lower < upper contract.
    return np.minimum(bounds[0], upper - 1e-6), np.maximum(bounds[1], lower + 1e-6)


def _attach_positions(body_q: np.ndarray, attach_frames, body_indices) -> np.ndarray:
    poses = np.asarray(body_q, dtype=np.float64)[:, list(body_indices)]
    offsets = np.asarray([frame.position for frame in attach_frames], dtype=np.float64)
    return poses[..., :3] + _quat_rotate(poses[..., 3:7], offsets)


def _attach_approach_directions(body_q: np.ndarray, attach_frames, body_indices) -> np.ndarray:
    return _attach_axis_directions(body_q, attach_frames, body_indices, (0., 0., 1.))


def _attach_axis_directions(body_q: np.ndarray, attach_frames, body_indices, axis) -> np.ndarray:
    poses = np.asarray(body_q, dtype=np.float64)[:, list(body_indices)]
    local_rotations = np.asarray([frame.rotation_xyzw for frame in attach_frames], dtype=np.float64)
    attach_rotations = _quat_multiply(poses[..., 3:7], local_rotations)
    return _quat_rotate(attach_rotations, np.asarray(axis))


def _task_space_residual(
    attach_positions: np.ndarray,
    openness: np.ndarray,
    approach_directions: np.ndarray,
    desired_attach_positions: np.ndarray,
    desired_openness: np.ndarray,
    desired_approach_directions: np.ndarray,
    pregrasp_mask: np.ndarray,
    *,
    up_directions: np.ndarray | None = None,
    desired_up_directions: np.ndarray | None = None,
) -> np.ndarray:
    positions = np.asarray(attach_positions, dtype=np.float64)
    target_positions = np.asarray(desired_attach_positions, dtype=np.float64)
    open_state = np.asarray(openness, dtype=np.float64)
    target_open = np.asarray(desired_openness, dtype=np.float64)
    approach = _normalize_directions(approach_directions)
    target_approach = _normalize_directions(desired_approach_directions)
    active = np.asarray(pregrasp_mask, dtype=bool)
    if (
        positions.shape[-2:] != (2, 3)
        or target_positions.shape != (2, 3)
        or open_state.shape != positions.shape[:-1]
        or target_open.shape != (2,)
        or approach.shape not in ((2, 3), positions.shape)
        or target_approach.shape != (2, 3)
        or active.shape != (2,)
    ):
        raise ValueError("task-space retarget values have invalid shapes")
    residual_shape = (*positions.shape[:-2], 6)
    position_residual = (
        (positions - target_positions).reshape(residual_shape) / _ATTACH_POSITION_LIMIT_M
    )
    approach_residual = (
        np.broadcast_to((approach - target_approach) * active[:, None], positions.shape)
    ).reshape(residual_shape) / PREGRASP_APPROACH_TOLERANCE_RAD
    gripper_residual = (open_state - target_open) / ATTACH_OPENNESS_THRESHOLD
    up_residual = np.zeros(residual_shape)
    if desired_up_directions is not None:
        up = _normalize_directions(up_directions)
        target_up = _normalize_directions(desired_up_directions)
        if up.shape not in ((2, 3), positions.shape) or target_up.shape != (2, 3):
            raise ValueError("up axes must have shape [2, 3]")
        up_residual = (
            np.broadcast_to((up - target_up) * active[:, None], positions.shape)
        ).reshape(residual_shape) / PREGRASP_APPROACH_TOLERANCE_RAD
    return np.concatenate(
        (position_residual, approach_residual, up_residual, gripper_residual), axis=-1,
    ).astype(np.float64)


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

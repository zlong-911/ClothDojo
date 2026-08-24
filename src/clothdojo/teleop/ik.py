"""Newton 1.5 dual-arm IK used by the XR control loop."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import warp as wp


@wp.kernel
def _select_finite_solution(
    candidate: wp.array(dtype=float),
    last_safe: wp.array(dtype=float),
    command: wp.array(dtype=float),
    status: wp.array(dtype=int),
):
    index = wp.tid()
    value = candidate[index]
    if not wp.isfinite(value):
        command[index] = last_safe[index]
        wp.atomic_max(status, 0, 2)
    else:
        last_safe[index] = value
        command[index] = value


@wp.kernel
def _gather_arm_command(
    source: wp.array(dtype=float),
    indices: wp.array(dtype=int),
    target: wp.array2d(dtype=float),
):
    index = wp.tid()
    target[0, index] = source[indices[index]]


@dataclass
class DualArmIKController:
    runtime: object
    iterations: int = 10
    use_cuda_graph: bool = True
    position_weight: float = 100.0
    rotation_weight: float = 1.0
    joint_limit_weight: float = 10.0
    solver: object = field(init=False, repr=False)

    def __post_init__(self) -> None:
        from newton.ik import (
            IKJacobianType,
            IKObjectiveJointLimit,
            IKObjectivePosition,
            IKObjectiveRotation,
            IKSolver,
        )

        if not isinstance(self.use_cuda_graph, bool):
            raise TypeError("use_cuda_graph must be a bool")
        self.model = self.runtime.model
        self.device = self.model.device
        self._solve_graph = None
        self.left_ee_body, self.right_ee_body = self.runtime.scene.end_effector_body_indices
        self.joint_coord_count = int(self.model.joint_coord_count)
        self._joint_q_in = wp.zeros(
            (1, self.joint_coord_count), dtype=wp.float32, device=self.device
        )
        self._joint_q_out = wp.zeros_like(self._joint_q_in)
        self._joint_q_in_flat = self._joint_q_in.reshape((self.joint_coord_count,))
        self._joint_q_out_flat = self._joint_q_out.reshape((self.joint_coord_count,))
        self._joint_q_safe = wp.clone(self.runtime.state_0.joint_q)
        self._joint_q_command = wp.clone(self.runtime.state_0.joint_q)
        self._solution_status = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._arm_indices = wp.array(
            self.runtime.scene.arm_joint_coord_indices,
            dtype=wp.int32,
            device=self.device,
        )
        self._arm_command = wp.zeros((1, 12), dtype=wp.float32, device=self.device)
        self.sync_from_state()

        left_pos, left_rot, right_pos, right_rot = self.measured_targets()
        self._left_target_pos = left_pos
        self._left_target_rot = left_rot
        self._right_target_pos = right_pos
        self._right_target_rot = right_rot
        self._left_positions = wp.array([left_pos], dtype=wp.vec3, device=self.device)
        self._right_positions = wp.array([right_pos], dtype=wp.vec3, device=self.device)
        self._left_rotations = wp.array([left_rot], dtype=wp.vec4, device=self.device)
        self._right_rotations = wp.array([right_rot], dtype=wp.vec4, device=self.device)
        objectives = [
            IKObjectivePosition(
                self.left_ee_body,
                wp.vec3(0.0),
                self._left_positions,
                float(self.position_weight),
            ),
            IKObjectiveRotation(
                self.left_ee_body,
                wp.quat_identity(),
                self._left_rotations,
                weight=float(self.rotation_weight),
            ),
            IKObjectivePosition(
                self.right_ee_body,
                wp.vec3(0.0),
                self._right_positions,
                float(self.position_weight),
            ),
            IKObjectiveRotation(
                self.right_ee_body,
                wp.quat_identity(),
                self._right_rotations,
                weight=float(self.rotation_weight),
            ),
            IKObjectiveJointLimit(
                self.model.joint_limit_lower,
                self.model.joint_limit_upper,
                float(self.joint_limit_weight),
            ),
        ]
        self.solver = IKSolver(
            model=self.model,
            n_problems=1,
            objectives=objectives,
            jacobian_mode=IKJacobianType.MIXED,
            lambda_initial=0.1,
        )

    def measured_targets(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        poses = self.runtime.state_0.body_q.numpy()
        left = np.asarray(poses[self.left_ee_body], dtype=np.float32).reshape(7)
        right = np.asarray(poses[self.right_ee_body], dtype=np.float32).reshape(7)
        return left[:3], _normalize_quat(left[3:]), right[:3], _normalize_quat(right[3:])

    def measured_pose_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        left_pos, left_rot, right_pos, right_rot = self.measured_targets()
        return _pose_matrix(left_pos, left_rot), _pose_matrix(right_pos, right_rot)

    def set_targets(self, left_pose: np.ndarray, right_pose: np.ndarray) -> None:
        self._left_target_pos = np.asarray(left_pose[:3, 3], dtype=np.float32)
        self._right_target_pos = np.asarray(right_pose[:3, 3], dtype=np.float32)
        self._left_target_rot = _matrix_to_quat(left_pose[:3, :3])
        self._right_target_rot = _matrix_to_quat(right_pose[:3, :3])
        self._left_positions.assign(self._left_target_pos.reshape(1, 3))
        self._right_positions.assign(self._right_target_pos.reshape(1, 3))
        self._left_rotations.assign(self._left_target_rot.reshape(1, 4))
        self._right_rotations.assign(self._right_target_rot.reshape(1, 4))

    def solve(self):
        if self.use_cuda_graph and wp.get_device(self.device).is_cuda:
            if self._solve_graph is None:
                self._capture_solve_graph()
            else:
                wp.capture_launch(self._solve_graph)
            return self._arm_command
        self._submit_solve()
        return self._arm_command

    @property
    def cuda_graph_enabled(self) -> bool:
        return self._solve_graph is not None

    def _submit_solve(self) -> None:
        self.sync_from_state()
        self.solver.step(self._joint_q_in, self._joint_q_out, iterations=int(self.iterations))
        self._solution_status.zero_()
        wp.launch(
            _select_finite_solution,
            dim=self.joint_coord_count,
            inputs=[
                self._joint_q_out_flat,
                self._joint_q_safe,
                self._joint_q_command,
                self._solution_status,
            ],
            device=self.device,
        )
        wp.launch(
            _gather_arm_command,
            dim=12,
            inputs=[self._joint_q_command, self._arm_indices],
            outputs=[self._arm_command],
            device=self.device,
        )

    def _capture_solve_graph(self) -> None:
        # Run once to compile Newton's specialized IK kernels. The result is
        # already valid for this call; later solves replay the captured graph.
        self._submit_solve()
        wp.synchronize_device(self.device)
        with wp.ScopedCapture(device=self.device, force_module_load=False) as capture:
            self._submit_solve()
        self._solve_graph = capture.graph

    def reset_from_state(self) -> None:
        self.solver.reset()
        self.sync_from_state()
        wp.copy(self._joint_q_safe, self.runtime.state_0.joint_q)
        wp.copy(self._joint_q_command, self.runtime.state_0.joint_q)
        left, right = self.measured_pose_matrices()
        self.set_targets(left, right)

    def solution_status(self) -> int:
        return int(self._solution_status.numpy()[0])

    def sync_from_state(self) -> None:
        wp.copy(self._joint_q_in_flat, self.runtime.state_0.joint_q)


def _normalize_quat(quaternion: np.ndarray) -> np.ndarray:
    value = np.asarray(quaternion, dtype=np.float32).reshape(4)
    norm = float(np.linalg.norm(value))
    if norm <= 1.0e-8:
        return np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float32)
    return value / norm


def _pose_matrix(position: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = _normalize_quat(quaternion)
    rotation = np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = position
    return result


def _matrix_to_quat(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        value = (
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
            0.25 * scale,
        )
        value = (value[0] / scale, value[1] / scale, value[2] / scale, value[3])
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            value = (
                0.25 * scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
            )
        elif index == 1:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            value = (
                (matrix[0, 1] + matrix[1, 0]) / scale,
                0.25 * scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
            )
        else:
            scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            value = (
                (matrix[0, 2] + matrix[2, 0]) / scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
                0.25 * scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            )
    return _normalize_quat(np.asarray(value, dtype=np.float32))

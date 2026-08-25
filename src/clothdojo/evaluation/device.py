"""Compact GPU-resident metric and success tracking for Newton batches."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from clothdojo.evaluation.metric import CanonicalTemplate, TorchFactorizedClothMetric
from clothdojo.evaluation.protocol import EvaluationProtocol


@dataclass(frozen=True, slots=True)
class EvaluationBatchResult:
    success: np.ndarray
    success_at_end: np.ndarray
    confirmed_step: np.ndarray
    stable_since_step: np.ndarray
    terminal_deformable_distance: np.ndarray
    terminal_rigid_distance: np.ndarray
    terminal_combined_distance: np.ndarray
    terminal_speed_rms_m_s: np.ndarray
    best_settled_step: np.ndarray
    best_settled_deformable_distance: np.ndarray
    best_settled_rigid_distance: np.ndarray
    best_settled_combined_distance: np.ndarray
    curve: np.ndarray


class NewtonEvaluationTracker:
    """Evaluate Newton's Warp arrays on their existing CUDA stream.

    The full particle state never leaves the device.  Only the compact result
    and scalar curve are copied once, after the rollout horizon.
    """

    def __init__(
        self,
        runtime,
        template: CanonicalTemplate,
        protocol: EvaluationProtocol,
    ) -> None:
        import torch
        import warp as wp

        self.runtime = runtime
        self.protocol = protocol
        self.batch_size = int(runtime.model.world_count)
        self.particle_count = template.particle_count
        ranges = runtime.grasp.particle_ranges()
        counts = {end - start for start, end in ranges}
        if counts != {template.particle_count}:
            raise ValueError(
                "canonical template particle count differs from the Newton batch"
            )
        if ranges != tuple(
            (world * template.particle_count, (world + 1) * template.particle_count)
            for world in range(self.batch_size)
        ):
            raise RuntimeError("evaluation requires contiguous homogeneous particle worlds")

        self._torch = torch
        self._warp = wp
        q = wp.to_torch(runtime.state_0.particle_q, requires_grad=False)
        qd = wp.to_torch(runtime.state_0.particle_qd, requires_grad=False)
        attached = wp.to_torch(runtime.grasp.attached, requires_grad=False)
        self._particle_q = q.reshape(self.batch_size, self.particle_count, 3)
        self._particle_qd = qd.reshape(self.batch_size, self.particle_count, 3)
        self._attached = attached.reshape(self.batch_size, 2)
        self._device = self._particle_q.device
        table_center_xy = np.asarray(runtime.spec.table_position[:2], dtype=np.float64)
        symmetry_center_xy = np.asarray(
            protocol.target.symmetry_center_xy,
            dtype=np.float64,
        )
        if not np.allclose(
            symmetry_center_xy,
            table_center_xy,
            rtol=0.0,
            atol=1.0e-9,
        ):
            raise ValueError(
                "protocol symmetry center differs from the runtime table center"
            )
        self._metric = TorchFactorizedClothMetric(
            template,
            device=self._device,
            deformable_weight=protocol.deformable_weight,
            symmetry=protocol.symmetry,
            symmetry_center_xy=symmetry_center_xy,
        )
        self._stream = None
        if self._particle_q.is_cuda:
            warp_stream = wp.get_stream(runtime.model.device)
            self._stream = torch.cuda.ExternalStream(
                warp_stream.cuda_stream,
                device=self._device,
            )

        self._success = torch.zeros(self.batch_size, dtype=torch.bool, device=self._device)
        self._streak = torch.zeros(self.batch_size, dtype=torch.int32, device=self._device)
        self._confirmed_step = torch.full(
            (self.batch_size,), -1, dtype=torch.int32, device=self._device
        )
        self._stable_since_step = torch.full_like(self._confirmed_step, -1)
        self._best_step = torch.full_like(self._confirmed_step, -1)
        self._best_combined = torch.full(
            (self.batch_size,), float("inf"), dtype=torch.float32, device=self._device
        )
        self._best_deformable = torch.full_like(self._best_combined, float("nan"))
        self._best_rigid = torch.full_like(self._best_combined, float("nan"))
        self._curve = torch.full(
            (protocol.sample_count, self.batch_size, 4),
            float("nan"),
            dtype=torch.float32,
            device=self._device,
        )
        self._last_step = -protocol.metric_stride
        self._terminal = None

    def record(self, step: int) -> bool:
        """Record a protocol sample; return false on non-sample control steps."""

        if step % self.protocol.metric_stride:
            return False
        if step != self._last_step + self.protocol.metric_stride:
            raise ValueError("evaluation samples must be recorded in protocol order")
        if step > self.protocol.horizon_steps:
            raise ValueError("evaluation sample exceeds the protocol horizon")
        context = (
            self._torch.cuda.stream(self._stream)
            if self._stream is not None
            else _NullContext()
        )
        with context:
            # Eager stepping may swap Newton's state objects. Refreshing these
            # wrappers is still zero-copy and always follows the live state.
            self._particle_q = self._warp.to_torch(
                self.runtime.state_0.particle_q, requires_grad=False
            ).reshape(self.batch_size, self.particle_count, 3)
            self._particle_qd = self._warp.to_torch(
                self.runtime.state_0.particle_qd, requires_grad=False
            ).reshape(self.batch_size, self.particle_count, 3)
            values = self._metric(self._particle_q, self._particle_qd)
            released = (self._attached == 0).all(dim=1)
            settled = values.speed_rms_m_s <= (
                self.protocol.particle_speed_rms_max_m_s
            )
            if self.protocol.require_released:
                settled &= released
            candidate = settled & (
                values.deformable_distance <= self.protocol.deformable_distance_max
            )
            if self.protocol.rigid_distance_max is not None:
                candidate &= values.rigid_distance <= self.protocol.rigid_distance_max
            self._streak = self._torch.where(
                candidate,
                self._streak + 1,
                self._torch.zeros_like(self._streak),
            )
            newly_confirmed = (~self._success) & (
                self._streak >= self.protocol.stable_samples
            )
            step_values = self._torch.full_like(self._confirmed_step, int(step))
            stable_values = self._torch.full_like(
                self._stable_since_step,
                int(
                    step
                    - (self.protocol.stable_samples - 1)
                    * self.protocol.metric_stride
                ),
            )
            self._confirmed_step = self._torch.where(
                newly_confirmed, step_values, self._confirmed_step
            )
            self._stable_since_step = self._torch.where(
                newly_confirmed, stable_values, self._stable_since_step
            )
            self._success |= newly_confirmed

            better = settled & (values.combined_distance < self._best_combined)
            self._best_step = self._torch.where(better, step_values, self._best_step)
            self._best_combined = self._torch.where(
                better, values.combined_distance, self._best_combined
            )
            self._best_deformable = self._torch.where(
                better, values.deformable_distance, self._best_deformable
            )
            self._best_rigid = self._torch.where(
                better, values.rigid_distance, self._best_rigid
            )
            slot = step // self.protocol.metric_stride
            self._curve[slot, :, 0] = values.deformable_distance
            self._curve[slot, :, 1] = values.rigid_distance
            self._curve[slot, :, 2] = values.combined_distance
            self._curve[slot, :, 3] = values.speed_rms_m_s
            self._terminal = values
        self._last_step = step
        return True

    def result(self) -> EvaluationBatchResult:
        if self._last_step != self.protocol.horizon_steps:
            raise RuntimeError("evaluation did not reach the protocol horizon")
        if self._terminal is None:
            raise RuntimeError("evaluation has no metric samples")
        if self._stream is not None:
            self._stream.synchronize()

        def host(value, *, dtype=None):
            result = value.detach().cpu().numpy()
            return result if dtype is None else result.astype(dtype, copy=False)

        terminal = self._terminal
        return EvaluationBatchResult(
            success=host(self._success, dtype=bool),
            success_at_end=host(
                self._streak >= self.protocol.stable_samples, dtype=bool
            ),
            confirmed_step=host(self._confirmed_step, dtype=np.int32),
            stable_since_step=host(self._stable_since_step, dtype=np.int32),
            terminal_deformable_distance=host(terminal.deformable_distance),
            terminal_rigid_distance=host(terminal.rigid_distance),
            terminal_combined_distance=host(terminal.combined_distance),
            terminal_speed_rms_m_s=host(terminal.speed_rms_m_s),
            best_settled_step=host(self._best_step, dtype=np.int32),
            best_settled_deformable_distance=host(self._best_deformable),
            best_settled_rigid_distance=host(self._best_rigid),
            best_settled_combined_distance=host(self._best_combined),
            curve=host(self._curve),
        )


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

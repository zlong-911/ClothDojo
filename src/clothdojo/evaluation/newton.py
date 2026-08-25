"""A no-trajectory Newton episode for high-throughput policy evaluation."""

from __future__ import annotations

from clothdojo.evaluation.device import EvaluationBatchResult, NewtonEvaluationTracker
from clothdojo.evaluation.metric import CanonicalTemplate
from clothdojo.evaluation.protocol import EvaluationProtocol
from clothdojo.rollout.newton import NewtonBatchEnvironment


class NewtonEvaluationEpisode(NewtonBatchEnvironment):
    """Render/infer in batches while retaining only compact metric scalars."""

    def __init__(
        self,
        runtime,
        *,
        observation_provider,
        template: CanonicalTemplate,
        protocol: EvaluationProtocol,
    ) -> None:
        if int(runtime.spec.control_fps) != protocol.control_fps:
            raise ValueError("evaluation and Newton control_fps differ")
        super().__init__(runtime, observation_provider=observation_provider)
        self.protocol = protocol
        self.tracker = NewtonEvaluationTracker(runtime, template, protocol)
        self.tracker.record(0)

    def step(self, policy_actions) -> None:
        super().step(policy_actions)
        self.tracker.record(self.sample_index)

    def result(self) -> EvaluationBatchResult:
        return self.tracker.result()

"""Policy-independent cloth evaluation primitives."""

from clothdojo.evaluation.metric import (
    CanonicalTemplate,
    FactorizedMetricValues,
    factorized_cloth_distance,
    load_canonical_template,
    save_canonical_template,
)
from clothdojo.evaluation.protocol import (
    CanonicalTargetProtocol,
    EvaluationProtocol,
    SuccessTracker,
    load_evaluation_protocol,
)
from clothdojo.evaluation.results import (
    EvaluationCaseKey,
    EvaluationCaseResult,
    load_evaluation_condition,
    summarize_paired_conditions,
    write_evaluation_cases,
)

__all__ = [
    "CanonicalTargetProtocol",
    "CanonicalTemplate",
    "EvaluationCaseKey",
    "EvaluationCaseResult",
    "EvaluationProtocol",
    "FactorizedMetricValues",
    "SuccessTracker",
    "factorized_cloth_distance",
    "load_canonical_template",
    "load_evaluation_condition",
    "load_evaluation_protocol",
    "save_canonical_template",
    "summarize_paired_conditions",
    "write_evaluation_cases",
]

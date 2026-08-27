"""Canonical full-evaluation report contracts and builders."""

from market_regime_engine.evaluation.report.contracts import (
    EVALUATION_REPORT_SCHEMA_VERSION,
    CandidateReport,
    CrossCandidateComparisonReport,
    EvaluationReport,
    FeatureSelectionReport,
    FoldReport,
    ReportIntegrity,
    ReportMetadata,
    canonical_json_bytes,
    canonical_payload_bytes,
    compute_report_payload_sha256,
    object_payload,
    seal_report,
    thaw_json,
    verify_report,
)

__all__ = [
    "EVALUATION_REPORT_SCHEMA_VERSION",
    "CandidateReport",
    "CrossCandidateComparisonReport",
    "EvaluationReport",
    "FeatureSelectionReport",
    "FoldReport",
    "ReportIntegrity",
    "ReportMetadata",
    "canonical_json_bytes",
    "canonical_payload_bytes",
    "compute_report_payload_sha256",
    "object_payload",
    "seal_report",
    "thaw_json",
    "verify_report",
]

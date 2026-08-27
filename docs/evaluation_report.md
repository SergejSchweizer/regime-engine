# EvaluationReport.v1

`EvaluationReport.v1` is the immutable, machine-readable audit record for one complete statistical evaluation parent run. It is evidence only: it never changes feature selection, HMM fitting, statistical champion ranking, model aliases, or downstream economic decisions.

## Completeness

A report is complete only when its configured candidate IDs exactly match the profile-version candidate universe and every candidate contains the exact ordered planned-fold IDs. Invalid folds and rejected candidates remain present with explicit failure evidence; unavailable metrics are `null`, never zero-filled, imputed, or interpolated.

## Canonical JSON

Serialization is UTF-8 JSON with sorted keys, compact separators, `ensure_ascii=True`, and `allow_nan=False`. UTC timestamps are normalized to RFC3339 `Z`. Canonical report evidence cannot contain raw source rows, credentials/DSNs, secret contents, or model-binary payloads.

The `integrity.report_payload_sha256` digest is SHA-256 over the canonical report payload with the complete `integrity` field excluded. `seal_report()` computes the digest, `verify_report()` recomputes it, and `canonical_json_bytes()` refuses to serialize an unsealed or mismatched report.

## Typed section boundaries

The contract separates:

- `ReportMetadata`: run/profile/source/build/plan/selection lineage;
- `FeatureSelectionReport`: frozen selected features and detailed selection evidence;
- `FoldReport`: one planned fold, including validity/failure evidence;
- `CandidateReport`: one configured candidate with all planned folds plus aggregate evidence;
- `CrossCandidateComparisonReport`: frozen champion/common-support comparison evidence;
- `ReportIntegrity`: canonical payload checksum;
- `EvaluationReport`: complete reconciled parent-run record.

Section-specific builders add the detailed statistical payloads in later implementation PRs without changing canonical identity or serialization semantics.

# Repository governance

The protected target is exactly `SergejSchweizer/regime-engine/main`.

Run `scripts/configure_github_governance.sh` with an authenticated GitHub CLI identity that has repository administration permission. The script is intentionally idempotent: it applies the target settings and then reads them back and fails unless every required invariant is present.

The target policy requires pull requests, strict `merge-gate` status checks, resolved review conversations, and administrator enforcement. Force pushes and branch deletion are disabled. Repository merge settings permit squash merge only, enable auto-merge, and delete merged head branches automatically.

The script never targets another repository or branch and does not weaken protection when a setting already exists.

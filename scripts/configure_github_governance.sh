#!/usr/bin/env bash
set -euo pipefail

readonly REPO="SergejSchweizer/regime-engine"
readonly BRANCH="main"

command -v gh >/dev/null || { echo "gh CLI is required" >&2; exit 2; }
command -v jq >/dev/null || { echo "jq is required" >&2; exit 2; }
gh auth status >/dev/null

permission="$(gh api "repos/${REPO}" --jq '.permissions.admin')"
if [[ "${permission}" != "true" ]]; then
  echo "authenticated GitHub identity must have admin permission on ${REPO}" >&2
  exit 3
fi

gh api --method PATCH "repos/${REPO}" \
  -F allow_squash_merge=true \
  -F allow_merge_commit=false \
  -F allow_rebase_merge=false \
  -F allow_auto_merge=true \
  -F delete_branch_on_merge=true >/dev/null

protection_json='{
  "required_status_checks": {"strict": true, "contexts": ["merge-gate"]},
  "enforce_admins": true,
  "required_pull_request_reviews": {
    "dismiss_stale_reviews": false,
    "require_code_owner_reviews": false,
    "required_approving_review_count": 0,
    "require_last_push_approval": false
  },
  "restrictions": null,
  "required_linear_history": false,
  "allow_force_pushes": false,
  "allow_deletions": false,
  "block_creations": false,
  "required_conversation_resolution": true,
  "lock_branch": false,
  "allow_fork_syncing": true
}'

gh api --method PUT "repos/${REPO}/branches/${BRANCH}/protection" \
  --input - <<<"${protection_json}" >/dev/null

repo_json="$(gh api "repos/${REPO}")"
protection="$(gh api "repos/${REPO}/branches/${BRANCH}/protection")"

jq -e '
  .allow_squash_merge == true and
  .allow_merge_commit == false and
  .allow_rebase_merge == false and
  .allow_auto_merge == true and
  .delete_branch_on_merge == true
' <<<"${repo_json}" >/dev/null

jq -e '
  .required_status_checks.strict == true and
  (.required_status_checks.contexts | index("merge-gate")) != null and
  .enforce_admins.enabled == true and
  .required_pull_request_reviews != null and
  .required_conversation_resolution.enabled == true and
  .allow_force_pushes.enabled == false and
  .allow_deletions.enabled == false
' <<<"${protection}" >/dev/null

echo "Verified governance for ${REPO}/${BRANCH}."

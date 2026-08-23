# Contributing and Git Policy

This repository uses a strict PR/branch/commit naming contract so work can be delegated safely to weak parallel agents and audited deterministically.

## Canonical PR name

Every PR has one machine-stable canonical name:

```text
PR-<three-digit-number>-<kebab-case-slug>
```

Example:

```text
PR-014-gaussian-hmm-adapter
```

The canonical PR name is the identifier that must appear unchanged in the PR title, branch name, and every commit subject belonging to that PR.

## PR title

PR titles must start with the exact canonical PR name followed by a colon and a short human-readable title:

```text
PR-014-gaussian-hmm-adapter: Implement configurable Gaussian HMM adapter
```

## Branch name

The branch name must be exactly:

```text
pr/<canonical-pr-name>
```

Example:

```text
pr/PR-014-gaussian-hmm-adapter
```

For legacy `BACKLOG.md` entries that currently show `pr/014-gaussian-hmm-adapter`, agents must normalize the branch to `pr/PR-014-gaussian-hmm-adapter`. The insertion of the `PR-` prefix is a repository-wide naming normalization only; it does not change the PR number, scope, dependencies, or allowed files.

## Conventional Commits

Every commit subject must follow Conventional Commits and use the exact canonical PR name as its scope:

```text
<type>(<canonical-pr-name>): <imperative description>
```

Examples:

```text
feat(PR-014-gaussian-hmm-adapter): implement full-covariance fitting
test(PR-014-gaussian-hmm-adapter): cover deterministic model reconstruction
docs(PR-014-gaussian-hmm-adapter): document Gaussian HMM parameters
```

Allowed commit types:

```text
feat
fix
docs
style
refactor
perf
test
build
ci
chore
revert
```

Breaking changes may use the standard Conventional Commits marker:

```text
feat(PR-014-gaussian-hmm-adapter)!: change fitted artifact contract
```

Generic subjects such as `update`, `changes`, `fix stuff`, or `WIP` are forbidden.

## Squash / merge commit

The final squash commit must also follow the same rule and contain the canonical PR name, for example:

```text
feat(PR-014-gaussian-hmm-adapter): implement configurable Gaussian HMM adapter
```

## Required Git status

Before creating the PR branch:

```text
git switch main
git pull --ff-only
git status --short
```

Required result:

```text
<empty output>
```

Before the final push, `git status --short` must again be empty after all intended files have been committed.

## Scope discipline

Agents must follow the PR's `Allowed files`, dependencies, and acceptance criteria from `BACKLOG.md`. For PR-045 through PR-050, and for the explicit PR-021/PR-022/PR-024/PR-035/PR-036 feature-selection addenda, agents must also read `BACKLOG_FEATURE_SELECTION.md`; those feature-selection sections are an additive backlog extension and take precedence over older conflicting feature-selection wording in `BACKLOG.md`. Agents must not broaden scope, refactor unrelated code, or edit another PR's files. `EVALUATION.md` must be updated in the same PR whenever the evaluation-sidecar rule in `BACKLOG.md` or `BACKLOG_FEATURE_SELECTION.md` applies.

For feature-selection work, the orchestrator should provide a weak agent only the single assigned PR section plus the shared statistical feature-selection contract from `BACKLOG_FEATURE_SELECTION.md`. An agent must stop if it would need an unmerged dependency, a file outside its declared scope, a different selection method/threshold, per-fold feature re-selection, HMM-based feature-subset search, or downstream ETF/portfolio data.

Any PR that creates or changes human-facing diagnostic plots must also satisfy `PLOT_STYLE.md`. This applies to MLflow fold-history plots, parent candidate-comparison plots, transition-matrix heatmaps, covariance heatmaps, and any future evaluation visualization even when the individual BACKLOG acceptance criteria do not repeat every rendering requirement.

The production feature-source boundary is governed by `DATA_SOURCE.md`. Any legacy backlog wording that describes direct upstream `regime-loader` Parquet as the production engine input must be interpreted according to `DATA_SOURCE.md`: production features are served by the `regime-loader` PostgreSQL replica at `10.10.1.3:54321`, while required CI remains hermetic.

## Precedence

For Git naming and commit-message conventions, this file is authoritative. For general implementation scope, dependency order, allowed files, and acceptance criteria, `BACKLOG.md` is authoritative. `BACKLOG_FEATURE_SELECTION.md` is authoritative for PR-045 through PR-050 and for its explicit feature-selection addenda to PR-021, PR-022, PR-024, PR-035, and PR-036; where those feature-selection sections conflict with older backlog wording, the extension wins. `DATA_SOURCE.md` is authoritative for upstream naming, production feature-source transport, PostgreSQL serving contract, lineage-snapshot semantics, and credential boundary. `EVALUATION.md` remains authoritative for implemented evaluation methodology and champion-selection semantics. `PLOT_STYLE.md` is authoritative for presentation quality, titles, legends, axis labels, state/candidate labeling, date-axis semantics, heatmap labeling, accessibility, export quality, and plot-manifest presentation metadata. Plot styling must never alter the statistical semantics defined by `EVALUATION.md`.

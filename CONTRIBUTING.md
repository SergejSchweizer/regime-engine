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
feat(PR-014-gaussian-hmm-adapter): implement diagonal covariance fitting
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

Agents must follow the PR's `Allowed files`, dependencies, and acceptance criteria from `BACKLOG.md`. They must not broaden scope, refactor unrelated code, or edit another PR's files. `EVALUATION.md` must be updated in the same PR whenever the evaluation-sidecar rule in `BACKLOG.md` applies.

## Precedence

For Git naming and commit-message conventions, this file is authoritative. For implementation scope, dependency order, allowed files, and acceptance criteria, `BACKLOG.md` is authoritative.

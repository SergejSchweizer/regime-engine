# Documentation Sidecar Policy

This repository treats **every Markdown (`*.md`) file as a documentation sidecar to the codebase**.

A Markdown sidecar is not an independent source of truth and must never describe an intended, historical, or aspirational implementation as if it were the current runtime behavior. The code, configuration, tests, schemas, workflows, and deployed contracts remain authoritative. Markdown documents exist to explain those artifacts accurately and to make them understandable to a new engineer or user.

## Mandatory rule

Every pull request that changes behavior, architecture, configuration, interfaces, data contracts, evaluation logic, model-selection logic, deployment, operations, or developer workflow **must update every affected Markdown sidecar in the same pull request**.

A Markdown file is stale when a reasonable reader could follow it and obtain a materially different understanding of the repository than they would obtain from the current codebase. Stale documentation is a defect.

This rule applies to all current and future Markdown files, including the README, architecture documents, backlogs, addenda, runbooks, implementation notes, onboarding guides, and topic-specific documentation.

## Sidecar requirements

Each Markdown document must:

1. **Reflect the current codebase.** Describe what the current default branch actually implements. Clearly label anything that is planned, deprecated, compatibility-only, experimental, or not yet implemented.
2. **Use code-aligned terminology.** Names of packages, modules, profiles, configuration keys, model families, evaluation IDs, paths, services, schemas, ports, aliases, and contracts must match the implementation.
3. **Avoid undocumented assumptions.** If behavior depends on a configuration value, code path, runtime dependency, upstream system, or external service, state that dependency explicitly.
4. **Remain internally consistent.** A change that makes one Markdown file contradict another requires the contradiction to be resolved in the same pull request.
5. **Be executable as guidance.** Commands, paths, examples, configuration snippets, and procedural steps must work against the current repository unless explicitly marked as illustrative.
6. **Onboard from zero context.** A new user must be able to understand the document's topic step by step without already knowing repository-specific vocabulary or hidden historical context.
7. **Lead from purpose to implementation.** Introduce the problem and role of the component before presenting implementation details.
8. **Explain boundaries.** State what the documented component owns, what it does not own, and which adjacent component or repository owns the next responsibility.
9. **Explain verification.** Where applicable, tell the reader how to verify that the described behavior is working, for example through tests, MLflow evidence, database state, CLI output, or CI checks.
10. **Preserve auditability.** Historical or superseded decisions may remain only when clearly marked as historical/superseded and when they cannot be confused with the current contract.

## Required onboarding shape

Topic-specific Markdown should normally guide a new reader through this sequence, adapting headings when necessary:

1. **What this is** — the component, workflow, model, or contract in simple terms.
2. **Why it exists** — the problem it solves and why the repository needs it.
3. **Where it fits** — upstream inputs, downstream consumers, and ownership boundaries.
4. **Core concepts** — define repository-specific terminology before relying on it.
5. **How it works** — describe the current end-to-end flow in execution order.
6. **Configuration and contracts** — identify the authoritative configuration, interfaces, schemas, and invariants.
7. **How to run or use it** — give reproducible steps from a clean checkout when applicable.
8. **How to verify it** — tests, expected evidence, observability, and failure signals.
9. **Failure modes and safeguards** — explain fail-closed behavior, invalid states, and recovery where relevant.
10. **Where to go next** — point to the next relevant sidecar, module, configuration file, or consumer.

The purpose is not to force identical headings into every document. The purpose is to ensure that a reader can progress from no repository knowledge to an accurate operational and conceptual understanding of the topic.

## Planning and backlog documents

Backlogs, addenda, and design notes are also sidecars and must distinguish state explicitly.

Use unambiguous language such as:

- `Implemented` / `Current behavior`
- `Planned` / `Not implemented`
- `Deprecated`
- `Superseded by ...`
- `Compatibility-only`

A PR title, backlog item, or design specification does **not** prove that functionality exists. Documentation must only describe functionality as implemented after the corresponding code and verification are present on the codebase being documented.

## Pull-request review checklist

Before merging a change, the author and reviewer should answer:

- Which Markdown sidecars are affected by this code/configuration change?
- Do they describe the new behavior exactly?
- Did the change introduce a contradiction with another Markdown file?
- Can a new user follow the affected topic from first principles through execution and verification?
- Are planned and implemented behaviors clearly separated?
- Do all commands, identifiers, versions, paths, ports, schemas, and examples still match the codebase?

If any answer is uncertain, the documentation update is part of the implementation and should be completed before merge.

## Definition of done

A repository change is complete only when:

- the implementation is correct,
- tests and quality gates pass,
- affected Markdown sidecars reflect the resulting codebase,
- a new user can follow the related documentation step by step,
- and no Markdown file presents superseded or future behavior as current behavior.

**Documentation drift is therefore treated as implementation drift, not as optional cleanup.**

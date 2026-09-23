# PR-511 Migration Map

This QA map records where retained material from the retired root sidecars now
lives. It is evidence for PR-512, not an additional contract owner.

| Retired section | Current owner | Decision |
| --- | --- | --- |
| PostgreSQL identity, lineage, credentials | `OPERATIONS.md` | retained and rewritten for `macro_loader.macro_features` |
| One-shot snapshot and interruption behavior | `OPERATIONS.md` | retained and rewritten without resume ledger |
| Plot presentation and MLflow diagnostics | `EVALUATION.md` | retained as the diagnostic artifact contract |
| MLflow lifecycle operations | `OPERATIONS.md` | retained as deployment/readback runbook |
| Historical legacy-removal narrative | `BACKLOG.md` history | intentionally superseded; no runtime instruction |

# Market Regime Engine — MLflow Plot Style Contract

Status date: 2026-08-23

This document is the normative rendering contract for every human-facing diagnostic plot produced by `market-regime-engine` and stored in MLflow. The statistical definitions remain governed by `EVALUATION.md`; this file governs presentation quality, labeling, visual consistency, accessibility, and export quality.

## Objective

Plots must be suitable for professional quantitative research review without requiring the operator to infer what an axis, line, state, candidate, date, unit, or matrix represents. A plot that is statistically correct but ambiguous, unlabeled, clipped, visually inconsistent, or difficult to compare across folds/candidates is incomplete.

## Global requirements

Every generated plot must:

- have a descriptive title that names the quantity being shown;
- identify the candidate/model context in either the title, subtitle/caption, or legend;
- have explicit x-axis and y-axis labels for all Cartesian plots;
- include units or scale semantics in the axis label whenever applicable, for example `Probability`, `Log likelihood per observation`, `Days`, `Fraction`, or `UTC test end`;
- use the actual fold `test_end` date for walk-forward trend x-axes;
- format dates in a compact, human-readable, timezone-consistent way;
- use a legend whenever one or more plotted series require identity, including candidate comparisons and per-state series;
- use stable legend labels based on canonical candidate IDs and persistent state IDs rather than raw library state numbers;
- use readable tick labels and prevent text overlap or clipping;
- use a restrained grid where it improves quantitative reading without dominating the data;
- use deterministic sizing, typography, line widths, marker sizes, ordering, and layout for reproducible artifacts;
- use a colorblind-safe, high-contrast deterministic series palette with stable candidate/state color assignment across all plots in one evaluation;
- remain understandable when printed or viewed on a light background and not rely on color alone to distinguish critical status;
- use markers and/or line style in addition to color where multiple series could otherwise be confused;
- never use 3-D effects, decorative gradients, chartjunk, misleading perspective, hidden secondary axes, or truncated categorical labels;
- never silently interpolate across invalid or missing folds;
- render invalid folds as explicit gaps and, where useful, a clearly labeled invalid marker;
- include only metrics with compatible units/scales on the same y-axis;
- use sensible y-axis padding so extrema are visible without excessive empty space;
- use a zero/reference line only when it has a real statistical interpretation;
- call `tight_layout` or an equivalent deterministic layout mechanism and save with a bounding box that prevents title, legend, tick, or axis-label clipping.

## Typography and hierarchy

The visual hierarchy must be consistent across all MLflow artifacts:

1. plot title: concise research question / metric name;
2. optional subtitle or caption: candidate ID, profile, fold/date context where required;
3. axis labels: metric semantics and units;
4. legend: candidate/state identity;
5. tick labels: readable but visually subordinate to axis labels;
6. annotations: only when they add analytical value.

Font sizes must be centrally configured in the plotting module rather than chosen ad hoc in individual plotting functions. Titles, axis labels, legends, and tick labels must remain readable in the MLflow artifact preview at normal desktop zoom.

## Output formats and resolution

Every required diagnostic plot must be exported as:

- PNG for immediate MLflow preview;
- SVG for publication-quality/vector inspection where the plot type is vector-compatible.

PNG output must use a deterministic minimum resolution of 180 DPI. Figure dimensions must be centrally defined and appropriate to the information density. Wide fold-history plots should use a landscape aspect ratio; square matrix heatmaps should use a near-square layout.

The plot manifest must record for each artifact:

- PNG path;
- SVG path when applicable;
- plot type;
- candidate ID;
- fold ID when applicable;
- source metric/artifact keys;
- x-axis field;
- x-axis label;
- y-axis label;
- legend entries;
- image dimensions;
- DPI for raster output;
- source artifact hash.

## Fold-history line plots

Every fold-history plot must:

- use chronological `test_end` dates on the x-axis;
- use a visible marker at each valid fold observation;
- draw connecting lines only between observations that are both valid under the documented missing-fold policy;
- display a legend identifying the candidate or persistent state for every series;
- use axis labels that state the statistical quantity and units;
- use concise date ticks with automatic spacing suitable for the number of folds;
- preserve the same candidate and state ordering across all related plots;
- make hard thresholds visible with a labeled reference line when the model profile declares a relevant threshold;
- annotate exceptional invalid/gate-failed folds only when the annotation remains readable and does not obscure neighboring data.

Examples of required labels:

- x-axis: `Test window end (UTC)`;
- OOS plot y-axis: `OOS predictive log likelihood per observation`;
- occupancy y-axis: `State occupancy fraction`;
- self-transition y-axis: `Self-transition probability`;
- duration y-axis: `Mean state duration (observations)` or the declared time unit;
- confidence y-axis: `Mean OOS confidence`;
- entropy y-axis: `Mean OOS entropy`;
- signature drift y-axis: `State-signature drift`.

## Parent-run candidate comparison plots

Cross-candidate plots must be immediately comparable across `gaussian_hmm_k2_full`, `gaussian_hmm_k3_full`, and `gaussian_hmm_k4_full`:

- one stable visual identity per candidate across every parent plot;
- legend labels must use the exact canonical candidate IDs;
- the legend order must be deterministic and follow K ascending unless the evaluation contract explicitly declares another order;
- all candidates must share the identical fold/date x-axis;
- no candidate may be shifted to conceal a missing/invalid fold;
- title must state that the plot is a walk-forward OOS candidate comparison;
- threshold/reference lines must have their own legend label or direct annotation.

## Per-state plots

Plots for occupancy, persistence, duration, or state drift must:

- use persistent state IDs, never raw HMM labels, in legends and annotations;
- keep the same state ordering and visual identity throughout the candidate run;
- include one legend entry per displayed state;
- make any aggregate summary line visually distinguishable and explicitly labeled, such as `minimum occupancy` or `maximum drift`.

## Transition-matrix heatmaps

Every transition heatmap must include:

- title containing candidate ID and fold ID/date context;
- x-axis label `Next state`;
- y-axis label `Current state`;
- persistent state IDs on both axes;
- a labeled color bar: `Transition probability`;
- a fixed probability scale from 0 to 1 so heatmaps are comparable across folds;
- cell-value annotations with a consistent numeric precision when matrix size remains legible;
- no raw library-state identifiers after state alignment.

## Full-covariance heatmaps

Every covariance heatmap must include:

- title containing candidate ID, fold ID/date context, and persistent state ID;
- x-axis label `Feature`;
- y-axis label `Feature`;
- exact ordered feature names on both axes;
- a labeled color bar: `Covariance`;
- all off-diagonal covariance terms;
- symmetric rendering around the matrix diagonal with square cells when practical;
- numeric cell annotations when feature dimensionality permits legible presentation;
- a documented fallback for high-dimensional matrices that suppresses cell text before suppressing axis labels;
- no per-fold auto-rescaling that makes two materially different covariance magnitudes look identical without disclosure. When a shared scale is used across related heatmaps, the scale bounds must be derived deterministically and recorded in the manifest.

## Labels, annotations, and precision

Numeric display precision must be chosen by metric family and configured centrally. The plotting implementation must avoid both meaningless precision and excessive rounding. Probability-like values should normally be displayed with enough precision to distinguish folds while remaining readable; log-likelihood and information criteria may use a separate precision rule.

Annotations must never obscure the plotted data. If annotations would overlap substantially, the plot must prefer a clean legend/tooltip-independent static representation rather than forcing every value onto the canvas.

## Professional quality gates

Plot generation must fail tests if any required plot is missing any of the following metadata/presentation elements:

- non-empty title;
- non-empty x-axis label for Cartesian plots;
- non-empty y-axis label for Cartesian plots;
- expected legend entries for multi-series plots;
- expected color-bar label for heatmaps;
- canonical candidate/persistent-state labels;
- real fold `test_end` mapping for fold-history plots;
- deterministic PNG artifact path;
- SVG counterpart where required;
- manifest entry with source metric/artifact lineage.

Tests should inspect Matplotlib figure/axes objects or the plotting specification before serialization rather than relying on fragile pixel-perfect screenshot comparisons.

## Relationship to other repository contracts

- `EVALUATION.md` is authoritative for what is measured and which plots/artifacts are required.
- `PLOT_STYLE.md` is authoritative for how those human-facing plots are rendered.
- `BACKLOG.md` defines implementation scope and PR ownership; any PR that creates or changes MLflow plots must satisfy this style contract even when individual visual details are not repeated in every acceptance checklist.
- Plot styling must never change model selection, metric values, fold validity, or statistical semantics.

# Human-factors pilot: de-identified aggregates (Sec. 4.5, Fig. 5)

Ethics approval GCU/RERC/2026/187 permits release of de-identified aggregate data only.
`aggregate.csv` holds the condition-level aggregates reported in Sec. 4.5; no
individual-level data are included or derivable.

| column | meaning |
|---|---|
| `metric` | `sus`, `decision_time_s`, `decision_accuracy` (proportion), `trust`, `confidence` |
| `condition` | `baseline` (no SHAP) or `shap` |
| `mean`, `sd`, `n` | condition aggregates (within-subjects, n = 20) |
| `p_value` | numeric upper bound used by the plotting script (on the `shap` row) |
| `p_reported` | p-value as reported in Sec. 4.5 |
| `cohens_d` | effect size (on the `shap` row) |

Trust: d = 0.82; confidence: d = 0.91.

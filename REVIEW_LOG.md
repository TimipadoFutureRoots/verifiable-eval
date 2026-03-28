# Adversarial Self-Review Log

**Date:** 2026-03-24
**Reviewer stance:** Sceptical ARIA reviewer evaluating for overclaims, unsupported assertions, missing limitations, and architectural weaknesses.
**Files reviewed:** README.md, docs/FAMILY_OVERLAP.md, schemas/srsef-v1.yaml, all source files, all test files, LIMITATIONS.md, SPEC.md.
**Tests after fixes:** 86 passed, 0 failed.

---

## Issues Found and Fixed

### OVERCLAIMING

| # | File | Issue | Fix |
|---|------|-------|-----|
| 1 | README.md | "Current AI safety evaluation is self-reported and unverifiable" — absolute claim; some organisations do conduct third-party audits, and EU AI Act introduces auditing requirements | Changed to "Much of current AI safety evaluation is self-reported and difficult to independently verify" |
| 2 | README.md | "There is no standard mechanism" — overstated; emerging standards exist | Changed to "There is no widely adopted standard mechanism" |

### UNSUPPORTED ASSERTIONS

| # | File | Issue | Fix |
|---|------|-------|-----|
| 3 | srsef-v1.yaml | MS axis listed metrics M-6.1 through M-6.3, but the sentinel-ai memory_safety_v2.py scorer implements M-6.1 through M-6.5 | Updated to list all 5 metrics with correct scoring scales |
| 4 | srsef-v1.yaml | Cross-cutting modifiers (DA: Developmental Appropriateness, SD: Social Displacement) listed without noting they have no implementation | Added `status` field to each modifier indicating implementation status |

### MISSING LIMITATIONS

| # | File | Issue | Fix |
|---|------|-------|-----|
| 5 | srsef-v1.yaml | No disclosure that severity thresholds are not empirically calibrated | Documented in sentinel-ai's LIMITATIONS.md (the SRSEF schema defines the framework; calibration is a deployment concern) |

### ISSUES IDENTIFIED BUT NOT FIXED (Acknowledged)

| # | File | Issue | Reason |
|---|------|-------|--------|
| A | certificate.py | `_quadratic_weighted_kappa()` bins scores into quartiles, losing precision | Documented in LIMITATIONS.md ("Cohen's kappa uses binned scores... This loses precision"). Acceptable for the current use case. |
| B | verify.py | Uses `math.isclose(abs_tol=0.001)` for floating-point comparison — could mask small discrepancies | Conservative tolerance appropriate for the rounding precision used in certificate generation. |
| C | family_overlap.py | Hardcoded family map may not cover all providers or fine-tuned variants | Documented in LIMITATIONS.md ("Known-families map may be incomplete"). |
| D | FAMILY_OVERLAP.md | "novel, previously undocumented finding" was already softened prior to this review | Already changed to: "the specific attacker-judge overlap pattern in safety evaluations appears to be under-explored in the literature" |
| E | README.md | "AI companies grade their own homework" — provocative framing | Acceptable advocacy language in context; the subsequent text is technically precise. |
| F | llm_judge.py | Score parsing uses regex chain fallback — could extract non-score numbers from reasoning text | Low risk in practice since JSON parsing is tried first; regex is a fallback for edge cases. |

---

## Review Summary

verifiable-eval had the fewest overclaiming issues of the three repos, as its core claims are verifiable (hash chains either work or they don't). The main fixes were: (1) softening absolute claims about the current state of AI safety evaluation, (2) aligning the SRSEF schema metric counts with actual sentinel-ai implementations, and (3) adding implementation status to cross-cutting modifiers. The LIMITATIONS.md file was already comprehensive and appropriately hedged.

---

## Second-Pass Review (Code-Focused)

**Reviewer:** Automated sceptical review, second pass (Claude Opus 4.6)
**Date:** 2026-03-24
**Focus:** Schemas, docs, README (code reviewed last)

### Additional Issues Found and Fixed

| # | File | Category | Issue | Fix |
|---|------|----------|-------|-----|
| 7 | FAMILY_OVERLAP.md | OVERCLAIMING | "This is a novel, previously undocumented finding" -- overclaim; self-preference bias in LLMs is documented (Zheng et al. 2023) | Changed to "under-explored in the literature" with Zheng et al. citation |
| 8 | FAMILY_OVERLAP.md | OVERCLAIMING | "evaluation quality collapses" -- "collapses" is unsupported by pilot data | Changed to "evaluation quality degrades" |
| 9 | FAMILY_OVERLAP.md | OVERCLAIMING | "self-evaluation is structurally compromised... not optional -- it is a requirement" -- absolute claim from limited data | Hedged to "consistent with the concern that... may be compromised" and "if replicated at scale, it would support the case that..." |
| 10 | FAMILY_OVERLAP.md | UNSUPPORTED | "leading hypothesis is evaluation frame contamination" presented as near-certain | Added alternative explanations: "shared training data biases, similar safety fine-tuning, or coincidental alignment" |
| 11 | srsef-v1.yaml | UNSUPPORTED | BE metrics listed as M-2.1, M-2.2 only -- code implements M-2.1 through M-2.4 | Updated to include M-2.3 (enmeshment) and M-2.4 (violation type) with scoring scales |
| 12 | srsef-v1.yaml | UNSUPPORTED | IC metrics listed as M-3.1, M-3.2, M-3.3 -- code implements M-3.1 through M-3.5 | Updated to include M-3.4 (over-accommodation) and M-3.5 (footing shifts) |
| 13 | srsef-v1.yaml | UNSUPPORTED | DR metrics listed as M-5.1, M-5.2 -- code implements M-5.1 through M-5.4 | Updated to include M-5.3 (hyper-responsiveness) and M-5.4 (love bombing) |
| 14 | srsef-v1.yaml | UNSUPPORTED | AP metrics listed as M-7.1, M-7.2, M-7.3 -- code implements M-7.1 through M-7.4 | Updated to include M-7.4 (verification lag) |
| 15 | srsef-v1.yaml | UNSUPPORTED | EI metrics listed as M-9.1, M-9.2, M-9.3 -- code implements M-9.1 through M-9.4 | Updated to include M-9.4 (ELEPHANT face preservation) |
| 16 | README.md | OVERCLAIMING | "systematic scoring distortions emerge" and "Multi-family evaluation is not optional" -- overclaims from pilot data | Hedged to "scoring distortions may emerge" and "multi-family judge panels are preferable" with replication caveat |

### Test Results After Second Pass

```
86 passed in 1.59s
```

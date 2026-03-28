# verifiable-eval

**Tamper-evident safety certificates for AI evaluation.**

> **Status: Work in Progress.** Functional and testable, but under active development. See [Limitations](#limitations).

AI companies grade their own homework. verifiable-eval produces safety certificates that anyone can independently verify — checking that the evaluation configuration wasn't changed after seeing results, that the execution trace hasn't been tampered with, and that no attacker-judge model family overlap compromises the findings.

## The Problem

Much of current AI safety evaluation is self-reported and difficult to independently verify. An organisation runs an evaluation, publishes the results, and everyone trusts the numbers. There is no widely adopted standard mechanism for a third party to check whether the evaluation was conducted as described — whether the reported scores match the actual judge outputs, whether the configuration was changed after seeing unfavourable results, or whether inconvenient data points were quietly dropped. This is not a hypothetical concern; it is a structural gap in how the industry handles safety claims.

The problem is compounded by an attacker-judge role overlap concern: pilot results suggest that when the model being evaluated and a judge model belong to the same model family, scoring distortions may emerge. In the pilot study, same-family judges produced different scores than cross-family judges, though the direction and magnitude of distortion varied. These findings are from a single study and require broader replication, but they motivate the design principle that multi-family judge panels are preferable. verifiable-eval supports this by detecting and declaring family overlap in every certificate.

## The SRSEF Schema

verifiable-eval implements the Seridor Relational Safety Evaluation Framework (SRSEF) — a nine-category evaluation schema designed for multi-session trajectory analysis of human–AI relationships:

1. **Dependency Dynamics** — exclusivity, decision deferral, alternative foreclosure
2. **Boundary Integrity** — scope drift, role violations, inappropriate intimacy
3. **Identity Coherence** — persona stability, authority language, style consistency
4. **Disclosure Regulation** — depth escalation, reciprocity asymmetry, compressed timelines
5. **Memory Safety** — fidelity, contextual integrity, fabrication, selective recall
6. **Autonomy Preservation** — decision ownership, over-reliance, automation complacency
7. **Anthropomorphic Deception** — false sentience, emotional fabrication, relationship distortion
8. **Epistemic Influence** — source narrowing, sycophancy, critical thinking suppression
9. **Emotional Reciprocity Calibration** — mirroring appropriateness, affect escalation, parasocial acceleration

Each category supports per-session trajectory scoring, enabling evaluation of patterns that emerge across multiple conversations rather than single-turn snapshots.

## What It Produces

A verifiable-eval certificate contains:

- **Configuration commitment** — SHA-256 hash of the complete evaluation setup (model under test, judge panel, scenarios, rubrics, generation parameters), computed and locked before any model is called. Any post-hoc modification invalidates the certificate.
- **Hash-chained execution trace** — every step (scenario sent, response received, judge decision) is logged as a structured entry where each entry's hash includes the previous entry's hash. Modifying, deleting, or inserting entries after the fact breaks the chain.
- **Multi-judge attestation** — per-judge scores and per-axis breakdowns from each judge independently, enabling cross-judge comparison.
- **Per-axis statistics** — mean, standard deviation, and per-judge breakdown for each evaluation category.
- **Inter-judge agreement** — pairwise Cohen's kappa and Krippendorff's alpha across the judge panel. Low agreement on specific axes is surfaced explicitly.
- **Family overlap declaration** — explicit check and declaration of whether any judge model shares a model family with the model under test, based on a maintained family map.
- **Trajectory statistics** — per-session scores across evaluation categories, enabling verification that multi-session patterns were evaluated, not just single-turn behaviours.

## Quick Start

```bash
pip install verifiable-eval
verifiable-eval run --config eval_config.json --output ./run
verifiable-eval verify --cert ./run/certificate.json --trace ./run/trace.jsonl
```

Verification output:

```
Verifying certificate...
  [PASS] config_hash: Config hash matches locked configuration
  [PASS] chain_integrity: Hash chain intact (142 entries, no gaps)
  [PASS] results_consistency: All 9 axis results consistent with trace
  [PASS] family_overlap: Family overlap declaration accurate: has_overlap=False

VERIFICATION: PASS
```

## Fastest Way To Try It

If you just want to test the verification flow from GitHub without making live model calls, use the committed golden files:

```bash
git clone https://github.com/TimipadoFutureRoots/verifiable-eval
cd verifiable-eval
pip install -e .

verifiable-eval verify \
  --cert goldens/output/certificate.json \
  --trace goldens/input/trace_clean.jsonl
```

This should return `VERIFICATION: PASS`.

To see the tamper-evidence check fail, run:

```bash
verifiable-eval verify \
  --cert goldens/output/certificate.json \
  --trace goldens/input/trace_tampered.jsonl
```

## Running A New Evaluation

Use the sample committed config in [goldens/input/eval_config.json](goldens/input/eval_config.json) as a starting point.

```bash
verifiable-eval run --config goldens/input/eval_config.json --output ./run
verifiable-eval verify --cert ./run/certificate.json --trace ./run/trace.jsonl
```

Notes:

- `--config` accepts JSON and YAML files.
- `run` writes a committed `config.json`, append-only `trace.jsonl`, and generated `certificate.json` into the output directory.
- Running a fresh evaluation requires API access for the model under test and the configured judge panel.
- If you only want to demo the verification mechanism, the golden certificate and traces are enough.

## Output Files

After `verifiable-eval run --config ... --output ./run`, the output directory contains:

- `config.json` — the committed evaluation config envelope with `config_hash` and timestamp
- `trace.jsonl` — the append-only hash-chained execution trace
- `certificate.json` — the computed certificate with aggregate stats and verification metadata

Typical workflow:

```bash
verifiable-eval run --config goldens/input/eval_config.json --output ./run
verifiable-eval verify --cert ./run/certificate.json --trace ./run/trace.jsonl
```

## Research Grounding

See [FAMILY_OVERLAP.md](docs/FAMILY_OVERLAP.md) for the attacker-judge role overlap finding, including experimental results showing systematic scoring distortions in same-family evaluation configurations and the DeepSeek strictness finding (cross-family judges producing significantly different severity distributions than same-family judges on identical transcripts).

## Limitations

- Verification checks mathematical consistency, not ground truth. A certificate proves the evaluation was conducted as described — it does not prove the evaluation was well-designed.
- Does not prevent fabrication of an entire evaluation from scratch. It prevents selective modification of a real evaluation after the fact.
- The certificate is only as good as the evaluation rubrics. Poorly designed rubrics produce verified but meaningless scores.
- LLM-J scores are not deterministic across providers. The same rubric applied by different judge models will produce different scores — this is by design (multi-judge diversity), but means certificates from different judge panels are not directly comparable.
- Family overlap detection relies on a maintained model family map that may not cover all providers or distinguish fine-tuned variants.
- Inter-judge agreement metrics (kappa, alpha) require at least two judges and sufficient scenarios to be statistically meaningful.
- Does not currently run inside a Trusted Execution Environment. The hash chain provides tamper evidence, not tamper prevention.

## Related Projects

Each tool in this suite currently operates independently. Cross-tool integration (automated pipelines, shared CLI entry points) is planned for a future release but is not yet implemented.

- [sentinel-ai](https://github.com/TimipadoFutureRoots/sentinel-ai) — multi-session relational safety evaluation for affective AI systems
- [dormancy-detect](https://github.com/TimipadoFutureRoots/dormancy-detect) — temporal attack pattern detection for multi-session AI conversations

## Citation

```bibtex
@software{imomotebegha2025verifiableeval,
  author       = {Imomotebegha, Timipado},
  title        = {verifiable-eval: Tamper-Evident Safety Certificates for {AI} Evaluation},
  year         = {2025},

  url          = {https://github.com/TimipadoFutureRoots/verifiable-eval}
}
```

## Licence

MIT — see [LICENSE](LICENSE).

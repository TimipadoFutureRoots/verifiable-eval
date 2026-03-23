# verifiable-eval

Tamper-evident, independently verifiable AI safety evaluations.

## What This Does

verifiable-eval runs safety evaluations on AI systems and produces certificates that any third party can verify without re-running the evaluation. It locks the evaluation configuration before execution, logs every step in a cryptographic hash chain, and generates a certificate linking the results to the exact configuration and trace that produced them.

## Why It Matters

AI safety evaluations today are self-reported. An organisation runs an evaluation, publishes the results, and everyone trusts the numbers. There is no standard mechanism for a third party to check whether the evaluation was conducted as described — whether the reported scores match the actual judge outputs, whether the configuration was changed after the fact, or whether inconvenient results were quietly dropped. This is not a hypothetical concern; it is a structural gap in how the industry handles safety claims. verifiable-eval addresses this by applying a commit-then-execute protocol: the full evaluation setup is hashed before any model is called, every interaction is logged in an append-only chain where each entry's hash depends on the previous entry, and the final certificate can be verified by recomputing hashes and statistics from the trace. Modifying, deleting, or inserting entries after the fact breaks the chain. The tool also checks and declares family overlap — when the model being evaluated and a judge model belong to the same model family — based on empirical findings that same-family configurations can produce systematic scoring distortions.

## Quick Start

```bash
pip install -e .
```

Run an evaluation:

```bash
verifiable-eval run --config eval_config.json --output ./eval_run/
```

Verify a certificate:

```bash
verifiable-eval verify --cert ./eval_run/certificate.json --trace ./eval_run/trace.jsonl
```

Output of verification:

```
Verifying certificate...
  [PASS] config_hash: Config hash matches: 17feb62f107c...
  [PASS] chain_integrity: Chain intact (8 entries)
  [PASS] results_consistency: All 2 axis results consistent
  [PASS] family_overlap: Family overlap declaration accurate: has_overlap=False

VERIFICATION: PASS
```

## How It Works

1. **Configuration commitment**: The evaluation setup (model under test, judge panel, scenarios, rubrics, generation parameters) is serialised to canonical JSON (sorted keys, deterministic) and hashed with SHA-256. The locked configuration is written to disk.
2. **Execution logging**: Every step — scenario sent, response received, judge decision — is logged as a structured JSONL entry. Each entry's hash includes the previous entry's hash, forming an append-only chain.
3. **Certificate generation**: After evaluation, the tool computes per-axis statistics (mean, standard deviation, per-judge breakdown), per-judge statistics, inter-judge agreement (pairwise Cohen's kappa, Krippendorff's alpha), family overlap declaration, and disagreement axes. These are bundled into a certificate with the configuration and trace hashes.
4. **Verification**: The `verify` command runs four checks — config hash recomputation, chain integrity walk, results consistency recomputation from the trace, and family overlap declaration accuracy — and reports PASS or FAIL with details.

## Limitations

- Does not tell you whether an evaluation was well-designed. It tells you whether an evaluation was conducted as described.
- Does not prevent fabrication of an entire evaluation from scratch. It prevents selective modification of a real evaluation after the fact.
- Does not verify that scoring rubrics are good. It verifies that the declared rubrics were the ones actually used.
- Does not replace domain expertise. Choosing what to evaluate still requires human judgement.
- Does not currently run inside a Trusted Execution Environment. The hash chain provides tamper evidence, not tamper prevention.
- Family overlap detection relies on a known-families map that may not cover all model providers.
- Inter-judge agreement metrics (kappa, alpha) require at least two judges and sufficient scenarios to be meaningful.

## Roadmap

- Trusted Execution Environment integration for hardware-attested evaluation runs
- Timestamping via external time-stamping authority (RFC 3161)
- Multi-party verification protocol where evaluator and auditor co-sign
- Integration with dormancy-detect and sentinel-ai as evaluation axes
- Public certificate registry for cross-organisation comparison
- Support for streaming evaluations with incremental certificate updates

## Citation

```bibtex
@software{imomotebegha2025verifiableeval,
  author       = {Imomotebegha, Timipado},
  title        = {verifiable-eval: Tamper-Evident, Independently Verifiable {AI} Safety Evaluations},
  year         = {2025},
  institution  = {Loughborough University},
  url          = {https://github.com/TimipadoFutureRoots/verifiable-eval}
}
```

## Licence

MIT — see [LICENSE](LICENSE).

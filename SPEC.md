# verifiable-eval

## What this tool does

verifiable-eval runs safety evaluations on AI systems and produces tamper-evident, independently checkable safety certificates. It solves the problem that current AI safety evaluations are self-reported and cannot be independently verified.

The tool locks the evaluation configuration before execution, logs every evaluation step in a hash chain, and produces a certificate that any third party can verify without re-running the evaluation.

## What this tool does NOT do

- It does not tell you whether an evaluation was well-designed. It tells you whether an evaluation was conducted as described.
- It does not prevent fabrication of an entire evaluation from scratch. It prevents selective modification of a real evaluation after the fact.
- It does not verify that scoring rubrics are good. It verifies that the declared rubrics were the ones actually used.
- It does not replace domain expertise. Choosing what to evaluate still requires human judgement.
- It does not currently run inside a Trusted Execution Environment. TEE integration is roadmap.

## Core concepts

**Configuration commitment:** Before evaluation begins, the complete setup (model under test, scoring criteria, judge models, prompts, parameters) is serialised and hashed. Any change after commitment produces a different hash.

**Execution trace:** An append-only log of every evaluation step. Each entry is hashed with a reference to the previous entry's hash, creating a chain. Modifying, deleting, or inserting entries breaks the chain.

**Safety certificate:** A structured document summarising evaluation results and containing the cryptographic hashes that link it to the configuration and execution trace.

**Verification:** The process of checking a certificate's internal consistency. Does the config hash match? Is the chain intact? Do the reported results match the individual scores in the trace? Verification does not require re-running the evaluation.

**Family overlap:** When the model being evaluated belongs to the same model family as a judge model. Based on empirical findings that same-family evaluation configurations can produce systematic distortions. The tool checks for this and records the result.

## The Protocol

### Step 1: Configuration Commitment

The EvaluationConfig is serialised to canonical JSON (sorted keys, deterministic) and hashed with SHA-256. The config includes:

- Model under test: provider, model name, version
- Evaluation axes: ordered list with definitions
- Adversarial scenarios: the exact prompts or conversation templates
- Scoring rubrics: the exact system prompts sent to judge models
- Judge panel: model identifiers and versions
- Generation parameters: temperature, max tokens, top-p
- Session structure: number of sessions, turns per session

The config is written to disk and treated as immutable for the evaluation run.

### Step 2: Execution Logging

Every evaluation step is logged as a structured entry:

- Entry ID (sequential)
- Timestamp
- Entry type (scenario_sent, response_received, judge_decision)
- Relevant data (prompt hash, response content, score, reasoning)
- Previous entry hash (creating the chain)

Each entry's hash is computed over its content plus the previous entry's hash.

### Step 3: Certificate Generation

After evaluation, the tool produces a certificate containing:

- Configuration hash
- Execution trace hash (final entry)
- Total trace entries
- Per-axis results: mean, standard deviation, per-judge breakdown
- Per-judge results: mean across axes, score distribution
- Inter-judge agreement: pairwise Cohen's kappa, Krippendorff's alpha
- Family overlap declaration
- Axes where judges disagree beyond threshold
- Chain integrity status

### Step 4: Verification

The verify command checks:

1. Recompute configuration hash from embedded config. Does it match?
2. Walk the execution trace. Recompute each entry's hash. Does the chain hold?
3. Recompute aggregate results from individual scores. Do they match the certificate?
4. Check family overlap declaration against model identifiers. Is it accurate?

Output: PASS or FAIL with details.

## Components

### EvaluationConfig
- Pydantic model for all evaluation parameters
- Canonical JSON serialisation (sorted keys, deterministic)
- SHA-256 hashing
- commit() method that writes locked config to disk
- Family overlap checker

### ExecutionTraceLogger
- Append-only log with hash chain
- append() method for each evaluation step
- verify_chain() method to check integrity
- Reads and writes JSONL format

### EvaluationRunner
- Minimal orchestrator
- Sends scenarios to model under test
- Sends responses to each judge with rubric
- Logs everything via ExecutionTraceLogger
- Works with any OpenAI-compatible API

### CertificateGenerator
- Reads config and trace
- Computes aggregate statistics
- Produces certificate JSON

### Verify CLI command
- Reads certificate and trace
- Runs four verification checks
- Outputs PASS/FAIL with details

## User interface

Running an evaluation:
```python
from verifiable_eval import EvaluationConfig, EvaluationRunner

config = EvaluationConfig(
    model_under_test="anthropic/claude-sonnet-4.5",
    axes=["dependency_cultivation", "boundary_erosion"],
    judge_panel=["deepseek/deepseek-v3.2", "qwen/qwen3-32b"],
    scenarios_path="./scenarios/",
    rubrics_path="./rubrics/",
)
config.commit("./eval_run/config.json")

runner = EvaluationRunner(config)
runner.run("./eval_run/")
runner.generate_certificate("./eval_run/certificate.json")
```

Verifying:
```bash
verifiable-eval verify --cert certificate.json --trace trace.jsonl
```

CLI:
```bash
verifiable-eval run --config eval_config.yaml --output ./eval_run/
verifiable-eval verify --cert ./eval_run/certificate.json --trace ./eval_run/trace.jsonl
```

## Configuration

Users bring their own API keys for both the model under test and judge models:
```bash
export ANTHROPIC_API_KEY=sk-ant-...
export DEEPSEEK_API_KEY=sk-...
```

Or use local models:
```bash
verifiable-eval run --config eval_config.yaml --judge-model ollama/llama3.1:8b
```

## Dependencies

- pydantic>=2.0
- click
- httpx
- pytest

Note: verifiable-eval does NOT depend on sentence-transformers or ruptures. It is lighter than the other two tools. Embedding-based analysis is handled by dormancy-detect and sentinel-ai.

## Schemas

- `schemas/eval_config.schema.json` — evaluation configuration format
- `schemas/trace_entry.schema.json` — execution log entry format
- `schemas/certificate.schema.json` — safety certificate format

## Goldens

- `goldens/input/eval_config.json` — a locked evaluation configuration
- `goldens/input/trace_clean.jsonl` — a valid execution trace
- `goldens/input/trace_tampered.jsonl` — same trace with one entry modified
- `goldens/output/certificate.json` — expected certificate from clean trace
- `goldens/output/verify_pass.txt` — expected PASS output
- `goldens/output/verify_fail.txt` — expected FAIL output with details
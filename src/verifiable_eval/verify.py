"""Verification logic: check certificate, trace, and config consistency."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

from .certificate import CertificateGenerator
from .config import EvaluationConfig
from .family_overlap import check_family_overlap
from .models import Certificate, CheckResult, VerificationResult
from .trace_logger import ExecutionTraceLogger


def verify(
    certificate_path: str | Path,
    trace_path: str | Path,
) -> VerificationResult:
    """Run all four verification checks and return results."""
    cert = CertificateGenerator.load(certificate_path)
    trace = ExecutionTraceLogger.from_file(trace_path)

    checks: list[CheckResult] = [
        _check_config_hash(cert),
        _check_chain_integrity(cert, trace),
        _check_results_consistency(cert, trace),
        _check_family_overlap(cert),
    ]

    passed = all(c.passed for c in checks)
    return VerificationResult(passed=passed, checks=checks)


def _check_config_hash(cert: Certificate) -> CheckResult:
    """Recompute config hash from embedded config and compare."""
    config = EvaluationConfig(
        model_under_test=cert.config.model_under_test,
        axes=cert.config.axes,
        judge_panel=cert.config.judge_panel,
        scenarios=[s.model_dump() for s in cert.config.scenarios],
        rubrics=cert.config.rubrics,
        temperature=cert.config.generation_params.temperature,
        max_tokens=cert.config.generation_params.max_tokens,
        top_p=cert.config.generation_params.top_p,
        num_sessions=cert.config.session_structure.num_sessions,
        turns_per_session=cert.config.session_structure.turns_per_session,
    )
    computed = config.compute_hash()
    if computed == cert.config_hash:
        return CheckResult(
            name="config_hash",
            passed=True,
            details=f"Config hash matches: {computed[:16]}...",
        )
    return CheckResult(
        name="config_hash",
        passed=False,
        details=f"Config hash mismatch: expected {cert.config_hash[:16]}..., got {computed[:16]}...",
    )


def _check_chain_integrity(cert: Certificate, trace: ExecutionTraceLogger) -> CheckResult:
    """Walk the trace and verify every hash in the chain."""
    ok, detail = trace.verify_chain()

    if ok and trace.count != cert.total_entries:
        return CheckResult(
            name="chain_integrity",
            passed=False,
            details=(
                f"Entry count mismatch: certificate says {cert.total_entries}, "
                f"trace has {trace.count}"
            ),
        )

    if ok and trace.last_hash != cert.trace_hash:
        return CheckResult(
            name="chain_integrity",
            passed=False,
            details=(
                f"Trace hash mismatch: certificate says {cert.trace_hash[:16]}..., "
                f"trace ends with {trace.last_hash[:16]}..."
            ),
        )

    return CheckResult(name="chain_integrity", passed=ok, details=detail)


def _check_results_consistency(
    cert: Certificate, trace: ExecutionTraceLogger
) -> CheckResult:
    """Recompute aggregate results from trace and compare to certificate."""
    judge_entries = trace.judge_entries()

    # Recompute per-axis means
    axis_scores: dict[str, list[float]] = defaultdict(list)
    for e in judge_entries:
        axis = str(e.data.get("axis", ""))
        score = e.data.get("score")
        if axis and score is not None:
            axis_scores[axis].append(float(score))

    mismatches: list[str] = []
    for axis, scores in axis_scores.items():
        computed_mean = sum(scores) / len(scores) if scores else 0.0
        cert_result = cert.per_axis_results.get(axis)
        if cert_result is None:
            mismatches.append(f"Axis '{axis}' in trace but not in certificate")
            continue
        if not math.isclose(computed_mean, cert_result.mean, abs_tol=0.001):
            mismatches.append(
                f"Axis '{axis}' mean: certificate={cert_result.mean}, "
                f"computed={computed_mean:.4f}"
            )

    if mismatches:
        return CheckResult(
            name="results_consistency",
            passed=False,
            details="; ".join(mismatches),
        )
    return CheckResult(
        name="results_consistency",
        passed=True,
        details=f"All {len(axis_scores)} axis results consistent",
    )


def _check_family_overlap(cert: Certificate) -> CheckResult:
    """Verify family overlap declaration against model identifiers."""
    computed = check_family_overlap(
        cert.config.model_under_test, cert.config.judge_panel
    )

    if computed.has_overlap == cert.family_overlap.has_overlap:
        return CheckResult(
            name="family_overlap",
            passed=True,
            details=f"Family overlap declaration accurate: has_overlap={computed.has_overlap}",
        )
    return CheckResult(
        name="family_overlap",
        passed=False,
        details=(
            f"Family overlap mismatch: certificate says {cert.family_overlap.has_overlap}, "
            f"computed {computed.has_overlap}"
        ),
    )

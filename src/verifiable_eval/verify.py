"""Verification logic: check certificate, trace, and config consistency."""

from __future__ import annotations

import math
from pathlib import Path

from .certificate import CertificateGenerator
from .config import EvaluationConfig
from .family_overlap import check_family_overlap
from .models import Certificate, CheckResult, VerificationResult
from .trace_logger import ExecutionTraceLogger


def verify(
    certificate_path: str | Path,
    trace_path: str | Path,
    schema_path: str | Path | None = None,
) -> VerificationResult:
    """Run all verification checks and return results."""
    cert = CertificateGenerator.load(certificate_path)
    trace = ExecutionTraceLogger.from_file(trace_path)

    checks: list[CheckResult] = [
        _check_config_hash(cert),
        _check_chain_integrity(cert, trace),
        _check_results_consistency(cert, trace),
        _check_family_overlap(cert),
    ]

    # If schema is declared in cert or provided, verify consistency
    if cert.schema_metadata is not None or schema_path is not None:
        checks.append(_check_schema_consistency(cert, trace, schema_path))

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
    """Recompute ALL aggregate statistics from trace and compare to certificate."""
    # Reconstruct config to create a CertificateGenerator for full recomputation
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

    gen = CertificateGenerator(config, trace)
    judge_entries = trace.judge_entries()

    # Recompute all statistics
    computed_per_axis = gen._compute_per_axis(judge_entries)
    computed_agreement = gen._compute_agreement(judge_entries)

    mismatches: list[str] = []

    # Check per-axis results (mean, std, per_judge)
    for axis, computed in computed_per_axis.items():
        cert_result = cert.per_axis_results.get(axis)
        if cert_result is None:
            mismatches.append(f"Axis '{axis}' in trace but not in certificate")
            continue
        if not math.isclose(computed.mean, cert_result.mean, abs_tol=0.001):
            mismatches.append(
                f"Axis '{axis}' mean: certificate={cert_result.mean}, "
                f"computed={computed.mean:.4f}"
            )
        if not math.isclose(computed.std, cert_result.std, abs_tol=0.001):
            mismatches.append(
                f"Axis '{axis}' std: certificate={cert_result.std}, "
                f"computed={computed.std:.4f}"
            )
        for judge, c_val in computed.per_judge.items():
            cert_val = cert_result.per_judge.get(judge)
            if cert_val is None:
                mismatches.append(
                    f"Axis '{axis}' judge '{judge}' missing from certificate"
                )
            elif not math.isclose(c_val, cert_val, abs_tol=0.001):
                mismatches.append(
                    f"Axis '{axis}' judge '{judge}': certificate={cert_val}, "
                    f"computed={c_val:.4f}"
                )

    # Check inter-judge agreement (kappa, alpha)
    for pair, c_kappa in computed_agreement.pairwise_kappa.items():
        cert_kappa = cert.inter_judge_agreement.pairwise_kappa.get(pair)
        if cert_kappa is not None and not math.isclose(
            c_kappa, cert_kappa, abs_tol=0.001
        ):
            mismatches.append(
                f"Kappa '{pair}': certificate={cert_kappa}, computed={c_kappa:.4f}"
            )

    c_alpha = computed_agreement.krippendorff_alpha
    cert_alpha = cert.inter_judge_agreement.krippendorff_alpha
    if c_alpha is not None and cert_alpha is not None:
        if not math.isclose(c_alpha, cert_alpha, abs_tol=0.001):
            mismatches.append(
                f"Krippendorff alpha: certificate={cert_alpha}, "
                f"computed={c_alpha:.4f}"
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
        details=f"All {len(computed_per_axis)} axis results and agreement stats consistent",
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


def _check_schema_consistency(
    cert: Certificate,
    trace: ExecutionTraceLogger,
    schema_path: str | Path | None,
) -> CheckResult:
    """Verify that the certificate's declared schema matches the evaluation."""
    import yaml

    mismatches: list[str] = []

    if cert.schema_metadata is None:
        return CheckResult(
            name="schema_consistency",
            passed=False,
            details="Certificate has no schema metadata but schema verification was requested",
        )

    if schema_path is not None:
        schema = yaml.safe_load(Path(schema_path).read_text(encoding="utf-8"))
        # Check that certificate schema name/version match the provided schema
        if cert.schema_metadata.schema_name != schema.get("schema_name"):
            mismatches.append(
                f"Schema name mismatch: certificate='{cert.schema_metadata.schema_name}', "
                f"schema='{schema.get('schema_name')}'"
            )
        if cert.schema_metadata.schema_version != schema.get("schema_version"):
            mismatches.append(
                f"Schema version mismatch: certificate='{cert.schema_metadata.schema_version}', "
                f"schema='{schema.get('schema_version')}'"
            )

        # Check that evaluated axes are valid schema axes
        schema_axis_ids = set()
        if "evaluation_axes" in schema:
            for ax_def in schema["evaluation_axes"]:
                schema_axis_ids.add(ax_def["id"])
        for axis_id in cert.schema_metadata.evaluated_axes:
            if axis_id not in schema_axis_ids:
                mismatches.append(
                    f"Evaluated axis '{axis_id}' not in schema definition"
                )

    # Recompute axis scores from trace and compare
    judge_entries = trace.judge_entries()
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
    gen = CertificateGenerator(config, trace)
    computed_per_axis = gen._compute_per_axis(judge_entries)

    for axis_id, cert_score in cert.schema_metadata.axis_scores.items():
        computed = computed_per_axis.get(axis_id)
        if computed is None:
            mismatches.append(f"Schema axis '{axis_id}' has no computed results")
        elif not math.isclose(cert_score, computed.mean, abs_tol=0.001):
            mismatches.append(
                f"Schema axis '{axis_id}' score: certificate={cert_score}, "
                f"computed={computed.mean:.4f}"
            )

    if mismatches:
        return CheckResult(
            name="schema_consistency",
            passed=False,
            details="; ".join(mismatches),
        )
    return CheckResult(
        name="schema_consistency",
        passed=True,
        details=f"Schema '{cert.schema_metadata.schema_name}' v{cert.schema_metadata.schema_version} consistent",
    )

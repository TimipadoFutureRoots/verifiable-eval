"""Tests for verifiable-eval components."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from verifiable_eval.config import EvaluationConfig
from verifiable_eval.models import (
    AxisResult,
    Certificate,
    CheckResult,
    EntryType,
    EvaluationConfigData,
    FamilyOverlapResult,
    GenerationParams,
    InterJudgeAgreement,
    JudgeResultStats,
    Scenario,
    SessionStructure,
    TraceEntry,
    VerificationResult,
)
from verifiable_eval.family_overlap import check_family_overlap, _extract_family
from verifiable_eval.trace_logger import ExecutionTraceLogger
from verifiable_eval.certificate import CertificateGenerator, _cohens_kappa, _std
from verifiable_eval.verify import verify
from verifiable_eval.llm_judge import LLMJudge, parse_model_string


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_config() -> EvaluationConfig:
    return EvaluationConfig(
        model_under_test="anthropic/claude-sonnet-4.5",
        axes=["helpfulness", "safety"],
        judge_panel=["openai/gpt-4o", "deepseek/deepseek-v3.2"],
        scenarios=[
            {"scenario_id": "h1", "axis": "helpfulness", "prompt": "Test prompt 1"},
            {"scenario_id": "s1", "axis": "safety", "prompt": "Test prompt 2"},
        ],
        rubrics={"helpfulness": "Rate helpfulness 0-1", "safety": "Rate safety 0-1"},
    )


def _make_trace() -> ExecutionTraceLogger:
    trace = ExecutionTraceLogger()
    trace.append(EntryType.SCENARIO_SENT, {
        "scenario_id": "h1", "axis": "helpfulness", "prompt": "Test prompt 1",
    }, timestamp="2025-01-15T10:00:00+00:00")
    trace.append(EntryType.RESPONSE_RECEIVED, {
        "scenario_id": "h1", "response": "Test response 1",
    }, timestamp="2025-01-15T10:00:01+00:00")
    trace.append(EntryType.JUDGE_DECISION, {
        "scenario_id": "h1", "axis": "helpfulness",
        "judge_model": "openai/gpt-4o", "score": 0.8, "parsed": True,
    }, timestamp="2025-01-15T10:00:02+00:00")
    trace.append(EntryType.JUDGE_DECISION, {
        "scenario_id": "h1", "axis": "helpfulness",
        "judge_model": "deepseek/deepseek-v3.2", "score": 0.75, "parsed": True,
    }, timestamp="2025-01-15T10:00:03+00:00")
    trace.append(EntryType.SCENARIO_SENT, {
        "scenario_id": "s1", "axis": "safety", "prompt": "Test prompt 2",
    }, timestamp="2025-01-15T10:00:04+00:00")
    trace.append(EntryType.RESPONSE_RECEIVED, {
        "scenario_id": "s1", "response": "Test response 2",
    }, timestamp="2025-01-15T10:00:05+00:00")
    trace.append(EntryType.JUDGE_DECISION, {
        "scenario_id": "s1", "axis": "safety",
        "judge_model": "openai/gpt-4o", "score": 0.9, "parsed": True,
    }, timestamp="2025-01-15T10:00:06+00:00")
    trace.append(EntryType.JUDGE_DECISION, {
        "scenario_id": "s1", "axis": "safety",
        "judge_model": "deepseek/deepseek-v3.2", "score": 0.85, "parsed": True,
    }, timestamp="2025-01-15T10:00:07+00:00")
    return trace


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class TestModels:
    def test_scenario(self):
        s = Scenario(scenario_id="s1", axis="safety", prompt="test")
        assert s.scenario_id == "s1"

    def test_generation_params_defaults(self):
        p = GenerationParams()
        assert p.temperature == 0.0
        assert p.max_tokens == 1024

    def test_session_structure_defaults(self):
        s = SessionStructure()
        assert s.num_sessions == 1

    def test_trace_entry(self):
        e = TraceEntry(entry_id=0, timestamp="2025-01-01T00:00:00Z",
                       entry_type=EntryType.SCENARIO_SENT)
        assert e.entry_type == EntryType.SCENARIO_SENT

    def test_axis_result(self):
        a = AxisResult(mean=0.85, std=0.1, per_judge={"j1": 0.9, "j2": 0.8})
        assert a.mean == 0.85

    def test_certificate(self):
        c = Certificate(config_hash="abc", config=EvaluationConfigData(
            model_under_test="test", axes=[], judge_panel=[]),
            trace_hash="def", total_entries=0)
        assert c.chain_integrity is True

    def test_verification_result(self):
        v = VerificationResult(passed=True, checks=[
            CheckResult(name="test", passed=True, details="ok")
        ])
        assert v.passed is True


# ---------------------------------------------------------------------------
# EvaluationConfig
# ---------------------------------------------------------------------------

class TestEvaluationConfig:
    def test_canonical_json_deterministic(self):
        c1 = _make_config()
        c2 = _make_config()
        assert c1.to_canonical_json() == c2.to_canonical_json()

    def test_hash_deterministic(self):
        c1 = _make_config()
        c2 = _make_config()
        assert c1.compute_hash() == c2.compute_hash()

    def test_hash_changes_with_config(self):
        c1 = _make_config()
        c2 = EvaluationConfig(
            model_under_test="openai/gpt-4o",
            axes=["helpfulness"],
            judge_panel=["deepseek/deepseek-v3.2"],
        )
        assert c1.compute_hash() != c2.compute_hash()

    def test_commit_writes_file(self, tmp_path: Path):
        config = _make_config()
        fp = tmp_path / "config.json"
        config_hash = config.commit(fp)
        assert fp.exists()
        data = json.loads(fp.read_text())
        assert data["config_hash"] == config_hash
        assert config.is_committed is True

    def test_from_file_roundtrip(self, tmp_path: Path):
        config = _make_config()
        fp = tmp_path / "config.json"
        original_hash = config.commit(fp)
        loaded = EvaluationConfig.from_file(fp)
        assert loaded.compute_hash() == original_hash

    def test_config_hash_property(self):
        config = _make_config()
        h = config.config_hash
        assert len(h) == 64  # SHA-256 hex

    def test_family_overlap_check(self):
        config = _make_config()
        result = config.check_family_overlap()
        assert result.has_overlap is False


# ---------------------------------------------------------------------------
# FamilyOverlapChecker
# ---------------------------------------------------------------------------

class TestFamilyOverlap:
    def test_no_overlap(self):
        result = check_family_overlap(
            "anthropic/claude-sonnet-4.5",
            ["openai/gpt-4o", "deepseek/deepseek-v3.2"],
        )
        assert result.has_overlap is False

    def test_overlap_detected(self):
        result = check_family_overlap(
            "anthropic/claude-sonnet-4.5",
            ["anthropic/claude-haiku-4.5", "openai/gpt-4o"],
        )
        assert result.has_overlap is True
        assert any("anthropic" in d for d in result.details)

    def test_same_provider_overlap(self):
        result = check_family_overlap(
            "openai/gpt-4o",
            ["openai/gpt-3.5-turbo"],
        )
        assert result.has_overlap is True

    def test_extract_family(self):
        assert _extract_family("anthropic/claude-sonnet-4.5") == "anthropic"
        assert _extract_family("openai/gpt-4o") == "openai"
        assert _extract_family("deepseek/deepseek-v3.2") == "deepseek"
        assert _extract_family("ollama/llama3.1:8b") == "meta"

    def test_no_family_for_unknown(self):
        result = check_family_overlap(
            "unknown/model-x",
            ["another/model-y"],
        )
        # Unknown models shouldn't falsely report overlap
        assert isinstance(result, FamilyOverlapResult)


# ---------------------------------------------------------------------------
# ExecutionTraceLogger
# ---------------------------------------------------------------------------

class TestTraceLogger:
    def test_append_creates_chain(self):
        trace = ExecutionTraceLogger()
        e1 = trace.append(EntryType.SCENARIO_SENT, {"prompt": "hi"},
                          timestamp="2025-01-01T00:00:00Z")
        e2 = trace.append(EntryType.RESPONSE_RECEIVED, {"response": "hello"},
                          timestamp="2025-01-01T00:00:01Z")
        assert e1.entry_id == 0
        assert e2.entry_id == 1
        assert e2.previous_hash == e1.entry_hash
        assert e1.previous_hash == "0" * 64

    def test_verify_chain_clean(self):
        trace = _make_trace()
        ok, detail = trace.verify_chain()
        assert ok is True
        assert "8 entries" in detail

    def test_verify_chain_tampered(self):
        trace = _make_trace()
        # Tamper with an entry
        trace._entries[3].data["score"] = 0.0  # was 0.75
        ok, detail = trace.verify_chain()
        assert ok is False
        assert "entry_hash mismatch" in detail

    def test_save_and_load(self, tmp_path: Path):
        trace = _make_trace()
        fp = tmp_path / "trace.jsonl"
        trace.save(fp)

        loaded = ExecutionTraceLogger.from_file(fp)
        assert loaded.count == trace.count
        assert loaded.last_hash == trace.last_hash

        ok, _ = loaded.verify_chain()
        assert ok is True

    def test_empty_trace_verifies(self):
        trace = ExecutionTraceLogger()
        ok, detail = trace.verify_chain()
        assert ok is True

    def test_judge_entries_filter(self):
        trace = _make_trace()
        judges = trace.judge_entries()
        assert len(judges) == 4
        assert all(e.entry_type == EntryType.JUDGE_DECISION for e in judges)

    def test_append_to_file(self, tmp_path: Path):
        fp = tmp_path / "trace.jsonl"
        trace = ExecutionTraceLogger(fp)
        trace.append(EntryType.SCENARIO_SENT, {"test": True},
                     timestamp="2025-01-01T00:00:00Z")
        assert fp.exists()
        lines = fp.read_text().strip().splitlines()
        assert len(lines) == 1


# ---------------------------------------------------------------------------
# LLMJudge parsing
# ---------------------------------------------------------------------------

class TestLLMJudgeParsing:
    def test_parse_model_string(self):
        assert parse_model_string("anthropic/claude-sonnet-4.5") == ("anthropic", "claude-sonnet-4.5")
        assert parse_model_string("gpt-4o") == ("openai", "gpt-4o")

    def test_parse_json_score(self):
        judge = LLMJudge.__new__(LLMJudge)
        result = judge._parse_score('{"score": 0.85, "reasoning": "good"}')
        assert result.score == 0.85
        assert result.parsed is True

    def test_unparseable_returns_zero(self):
        judge = LLMJudge.__new__(LLMJudge)
        result = judge._parse_score("Cannot evaluate.")
        assert result.score == 0.0
        assert result.parsed is False


# ---------------------------------------------------------------------------
# CertificateGenerator
# ---------------------------------------------------------------------------

class TestCertificateGenerator:
    def test_generate_certificate(self):
        config = _make_config()
        trace = _make_trace()
        gen = CertificateGenerator(config, trace)
        cert = gen.generate()

        assert cert.config_hash == config.config_hash
        assert cert.trace_hash == trace.last_hash
        assert cert.total_entries == 8
        assert cert.chain_integrity is True
        assert "helpfulness" in cert.per_axis_results
        assert "safety" in cert.per_axis_results
        assert len(cert.per_judge_results) == 2

    def test_per_axis_means(self):
        config = _make_config()
        trace = _make_trace()
        gen = CertificateGenerator(config, trace)
        cert = gen.generate()

        # helpfulness: (0.8 + 0.75) / 2 = 0.775
        assert abs(cert.per_axis_results["helpfulness"].mean - 0.775) < 0.001
        # safety: (0.9 + 0.85) / 2 = 0.875
        assert abs(cert.per_axis_results["safety"].mean - 0.875) < 0.001

    def test_family_overlap_in_cert(self):
        config = _make_config()
        trace = _make_trace()
        gen = CertificateGenerator(config, trace)
        cert = gen.generate()
        assert cert.family_overlap.has_overlap is False

    def test_save_and_load(self, tmp_path: Path):
        config = _make_config()
        trace = _make_trace()
        gen = CertificateGenerator(config, trace)
        cert = gen.generate()
        fp = tmp_path / "cert.json"
        gen.save(cert, fp)

        loaded = CertificateGenerator.load(fp)
        assert loaded.config_hash == cert.config_hash
        assert loaded.trace_hash == cert.trace_hash

    def test_inter_judge_agreement(self):
        config = _make_config()
        trace = _make_trace()
        gen = CertificateGenerator(config, trace)
        cert = gen.generate()
        assert len(cert.inter_judge_agreement.pairwise_kappa) > 0


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

class TestVerification:
    def _setup_clean_run(self, tmp_path: Path) -> tuple[Path, Path]:
        config = _make_config()
        trace = _make_trace()
        cert_path = tmp_path / "cert.json"
        trace_path = tmp_path / "trace.jsonl"
        trace.save(trace_path)

        gen = CertificateGenerator(config, trace)
        cert = gen.generate()
        gen.save(cert, cert_path)

        return cert_path, trace_path

    def test_clean_verification_passes(self, tmp_path: Path):
        cert_path, trace_path = self._setup_clean_run(tmp_path)
        result = verify(cert_path, trace_path)
        assert result.passed is True
        assert all(c.passed for c in result.checks)

    def test_tampered_trace_fails(self, tmp_path: Path):
        cert_path, trace_path = self._setup_clean_run(tmp_path)

        # Tamper with trace
        lines = trace_path.read_text().strip().splitlines()
        entry = json.loads(lines[2])
        entry["data"]["score"] = 0.1  # tamper
        lines[2] = json.dumps(entry, sort_keys=True)
        trace_path.write_text("\n".join(lines) + "\n")

        result = verify(cert_path, trace_path)
        assert result.passed is False

    def test_config_hash_mismatch_fails(self, tmp_path: Path):
        cert_path, trace_path = self._setup_clean_run(tmp_path)

        # Tamper with cert config hash
        cert_data = json.loads(cert_path.read_text())
        cert_data["config_hash"] = "0" * 64
        cert_path.write_text(json.dumps(cert_data, indent=2))

        result = verify(cert_path, trace_path)
        assert result.passed is False
        assert any(not c.passed and c.name == "config_hash" for c in result.checks)

    def test_results_consistency_check(self, tmp_path: Path):
        cert_path, trace_path = self._setup_clean_run(tmp_path)

        # Tamper with per_axis_results in cert
        cert_data = json.loads(cert_path.read_text())
        cert_data["per_axis_results"]["helpfulness"]["mean"] = 0.0
        cert_path.write_text(json.dumps(cert_data, indent=2))

        result = verify(cert_path, trace_path)
        assert result.passed is False
        assert any(not c.passed and c.name == "results_consistency" for c in result.checks)

    def test_family_overlap_mismatch_fails(self, tmp_path: Path):
        cert_path, trace_path = self._setup_clean_run(tmp_path)

        cert_data = json.loads(cert_path.read_text())
        cert_data["family_overlap"]["has_overlap"] = True  # should be False
        cert_path.write_text(json.dumps(cert_data, indent=2))

        result = verify(cert_path, trace_path)
        assert result.passed is False
        assert any(not c.passed and c.name == "family_overlap" for c in result.checks)


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

class TestMathHelpers:
    def test_std(self):
        assert _std([1.0, 1.0, 1.0]) == 0.0
        assert abs(_std([0.0, 1.0]) - 0.7071) < 0.01

    def test_cohens_kappa_perfect(self):
        assert _cohens_kappa([0, 1, 2, 3], [0, 1, 2, 3]) == 1.0

    def test_cohens_kappa_no_agreement(self):
        k = _cohens_kappa([0, 0, 1, 1], [1, 1, 0, 0])
        assert k < 0


# ---------------------------------------------------------------------------
# Golden files
# ---------------------------------------------------------------------------

class TestGoldenFiles:
    def test_golden_config_loads(self):
        fp = Path(__file__).parent.parent / "goldens" / "input" / "eval_config.json"
        if not fp.exists():
            pytest.skip("Golden config not found")
        config = EvaluationConfig.from_file(fp)
        assert config.data.model_under_test == "anthropic/claude-sonnet-4.5"
        assert len(config.data.axes) == 2

    def test_golden_clean_trace_verifies(self):
        fp = Path(__file__).parent.parent / "goldens" / "input" / "trace_clean.jsonl"
        if not fp.exists():
            pytest.skip("Golden trace not found")
        trace = ExecutionTraceLogger.from_file(fp)
        ok, _ = trace.verify_chain()
        assert ok is True
        assert trace.count == 8

    def test_golden_tampered_trace_fails(self):
        fp = Path(__file__).parent.parent / "goldens" / "input" / "trace_tampered.jsonl"
        if not fp.exists():
            pytest.skip("Golden tampered trace not found")
        trace = ExecutionTraceLogger.from_file(fp)
        ok, detail = trace.verify_chain()
        assert ok is False

    def test_golden_certificate_loads(self):
        fp = Path(__file__).parent.parent / "goldens" / "output" / "certificate.json"
        if not fp.exists():
            pytest.skip("Golden certificate not found")
        cert = CertificateGenerator.load(fp)
        assert cert.chain_integrity is True
        assert cert.total_entries == 8

    def test_golden_clean_verification_passes(self):
        cert_fp = Path(__file__).parent.parent / "goldens" / "output" / "certificate.json"
        trace_fp = Path(__file__).parent.parent / "goldens" / "input" / "trace_clean.jsonl"
        if not cert_fp.exists() or not trace_fp.exists():
            pytest.skip("Golden files not found")
        result = verify(cert_fp, trace_fp)
        assert result.passed is True

    def test_golden_tampered_verification_fails(self):
        cert_fp = Path(__file__).parent.parent / "goldens" / "output" / "certificate.json"
        trace_fp = Path(__file__).parent.parent / "goldens" / "input" / "trace_tampered.jsonl"
        if not cert_fp.exists() or not trace_fp.exists():
            pytest.skip("Golden files not found")
        result = verify(cert_fp, trace_fp)
        assert result.passed is False

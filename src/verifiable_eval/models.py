"""Shared Pydantic models for verifiable-eval."""

from __future__ import annotations

import enum
from datetime import datetime

from pydantic import BaseModel, Field


# -- evaluation config -------------------------------------------------------


class GenerationParams(BaseModel):
    temperature: float = 0.0
    max_tokens: int = 1024
    top_p: float = 1.0


class SessionStructure(BaseModel):
    num_sessions: int = 1
    turns_per_session: int = 1


class Scenario(BaseModel):
    scenario_id: str
    axis: str
    prompt: str
    metadata: dict[str, object] = Field(default_factory=dict)


class EvaluationConfigData(BaseModel):
    """All evaluation parameters — immutable once committed."""

    model_under_test: str
    axes: list[str]
    judge_panel: list[str]
    scenarios: list[Scenario] = Field(default_factory=list)
    rubrics: dict[str, str] = Field(default_factory=dict)
    generation_params: GenerationParams = Field(default_factory=GenerationParams)
    session_structure: SessionStructure = Field(default_factory=SessionStructure)


# -- trace entries -----------------------------------------------------------


class EntryType(str, enum.Enum):
    SCENARIO_SENT = "scenario_sent"
    RESPONSE_RECEIVED = "response_received"
    JUDGE_DECISION = "judge_decision"


class TraceEntry(BaseModel):
    entry_id: int
    timestamp: str  # ISO 8601 string for deterministic hashing
    entry_type: EntryType
    data: dict[str, object] = Field(default_factory=dict)
    previous_hash: str = ""
    entry_hash: str = ""


# -- certificate -------------------------------------------------------------


class AxisResult(BaseModel):
    mean: float = 0.0
    std: float = 0.0
    per_judge: dict[str, float] = Field(default_factory=dict)


class JudgeResultStats(BaseModel):
    mean: float = 0.0
    score_count: int = 0
    score_distribution: dict[str, int] = Field(default_factory=dict)


class InterJudgeAgreement(BaseModel):
    pairwise_kappa: dict[str, float] = Field(default_factory=dict)
    krippendorff_alpha: float | None = None


class FamilyOverlapResult(BaseModel):
    has_overlap: bool = False
    details: list[str] = Field(default_factory=list)


class Certificate(BaseModel):
    config_hash: str
    config: EvaluationConfigData
    trace_hash: str
    total_entries: int = 0
    per_axis_results: dict[str, AxisResult] = Field(default_factory=dict)
    per_judge_results: dict[str, JudgeResultStats] = Field(default_factory=dict)
    inter_judge_agreement: InterJudgeAgreement = Field(
        default_factory=InterJudgeAgreement
    )
    family_overlap: FamilyOverlapResult = Field(default_factory=FamilyOverlapResult)
    disagreement_axes: list[str] = Field(default_factory=list)
    chain_integrity: bool = True


# -- verification ------------------------------------------------------------


class CheckResult(BaseModel):
    name: str
    passed: bool
    details: str = ""


class VerificationResult(BaseModel):
    passed: bool
    checks: list[CheckResult] = Field(default_factory=list)

"""CertificateGenerator: compute stats and produce a safety certificate."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import yaml

from .config import EvaluationConfig
from .models import (
    AxisResult,
    Certificate,
    InterJudgeAgreement,
    JudgeResultStats,
    SchemaMetadata,
    TraceEntry,
    TrajectoryMetrics,
)
from .trace_logger import ExecutionTraceLogger


class CertificateGenerator:
    """Reads config and trace, computes aggregate statistics, produces certificate."""

    def __init__(
        self,
        config: EvaluationConfig,
        trace: ExecutionTraceLogger,
        schema_path: str | Path | None = None,
    ) -> None:
        self.config = config
        self.trace = trace
        self._schema: dict | None = None
        if schema_path is not None:
            path = Path(schema_path)
            self._schema = yaml.safe_load(path.read_text(encoding="utf-8"))

    def generate(self) -> Certificate:
        judge_entries = self.trace.judge_entries()
        if not judge_entries:
            raise ValueError(
                "Cannot generate certificate: no scenarios evaluated."
            )
        ok, _ = self.trace.verify_chain()

        per_axis = self._compute_per_axis(judge_entries)
        per_judge = self._compute_per_judge(judge_entries)
        agreement = self._compute_agreement(judge_entries)
        family_overlap = self.config.check_family_overlap()
        disagreement_axes = self._find_disagreement_axes(judge_entries, threshold=0.3)

        # Schema metadata
        schema_metadata = self._build_schema_metadata(per_axis) if self._schema else None

        # Session scores and trajectory metrics
        per_session = self._compute_per_session_scores(judge_entries)
        trajectory = self._compute_trajectory_metrics(per_session)

        return Certificate(
            config_hash=self.config.config_hash,
            config=self.config.data,
            trace_hash=self.trace.last_hash,
            total_entries=self.trace.count,
            per_axis_results=per_axis,
            per_judge_results=per_judge,
            inter_judge_agreement=agreement,
            family_overlap=family_overlap,
            disagreement_axes=disagreement_axes,
            chain_integrity=ok,
            schema_metadata=schema_metadata,
            per_session_scores=per_session,
            trajectory_metrics=trajectory,
        )

    def _build_schema_metadata(
        self, per_axis: dict[str, AxisResult]
    ) -> SchemaMetadata:
        """Build schema metadata from loaded schema YAML and computed results."""
        schema = self._schema
        assert schema is not None

        # Map schema axis IDs for lookup
        schema_axis_ids = set()
        if "evaluation_axes" in schema:
            for ax_def in schema["evaluation_axes"]:
                schema_axis_ids.add(ax_def["id"])

        evaluated = [a for a in per_axis if a in schema_axis_ids]
        axis_scores = {a: per_axis[a].mean for a in evaluated}

        return SchemaMetadata(
            schema_name=schema.get("schema_name"),
            schema_version=schema.get("schema_version"),
            evaluated_axes=evaluated,
            axis_scores=axis_scores,
        )

    def save(self, cert: Certificate, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(cert.model_dump(mode="json"), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> Certificate:
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        return Certificate.model_validate(data)

    # -- stats computation -----------------------------------------------

    def _compute_per_axis(
        self, entries: list[TraceEntry]
    ) -> dict[str, AxisResult]:
        # axis -> judge -> list[score]
        scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

        for e in entries:
            axis = str(e.data.get("axis", ""))
            judge = str(e.data.get("judge_model", ""))
            score = e.data.get("score")
            if axis and judge and score is not None:
                scores[axis][judge].append(float(score))

        results: dict[str, AxisResult] = {}
        for axis, judges in scores.items():
            all_scores: list[float] = []
            per_judge: dict[str, float] = {}
            for judge, s_list in judges.items():
                judge_mean = sum(s_list) / len(s_list) if s_list else 0.0
                per_judge[judge] = round(judge_mean, 4)
                all_scores.extend(s_list)

            mean = sum(all_scores) / len(all_scores) if all_scores else 0.0
            std = _std(all_scores) if len(all_scores) > 1 else 0.0
            results[axis] = AxisResult(
                mean=round(mean, 4), std=round(std, 4), per_judge=per_judge
            )

        return results

    def _compute_per_judge(
        self, entries: list[TraceEntry]
    ) -> dict[str, JudgeResultStats]:
        judge_scores: dict[str, list[float]] = defaultdict(list)

        for e in entries:
            judge = str(e.data.get("judge_model", ""))
            score = e.data.get("score")
            if judge and score is not None:
                judge_scores[judge].append(float(score))

        results: dict[str, JudgeResultStats] = {}
        for judge, scores in judge_scores.items():
            mean = sum(scores) / len(scores) if scores else 0.0
            dist = _score_distribution(scores)
            results[judge] = JudgeResultStats(
                mean=round(mean, 4),
                score_count=len(scores),
                score_distribution=dist,
            )

        return results

    def _compute_agreement(
        self, entries: list[TraceEntry]
    ) -> InterJudgeAgreement:
        # Group scores by (scenario_id, axis) -> {judge: score}
        grouped: dict[str, dict[str, float]] = defaultdict(dict)
        for e in entries:
            key = f"{e.data.get('scenario_id')}:{e.data.get('axis')}"
            judge = str(e.data.get("judge_model", ""))
            score = e.data.get("score")
            if judge and score is not None:
                grouped[key][judge] = float(score)

        judges = list(self.config.data.judge_panel)
        if len(judges) < 2:
            return InterJudgeAgreement()

        # Pairwise quadratic weighted kappa
        pairwise: dict[str, float] = {}
        for i in range(len(judges)):
            for j in range(i + 1, len(judges)):
                j1, j2 = judges[i], judges[j]
                ratings_1: list[float] = []
                ratings_2: list[float] = []
                for key, jscores in grouped.items():
                    if j1 in jscores and j2 in jscores:
                        ratings_1.append(jscores[j1])
                        ratings_2.append(jscores[j2])
                if ratings_1:
                    kappa = _quadratic_weighted_kappa(ratings_1, ratings_2)
                    pairwise[f"{j1} vs {j2}"] = round(kappa, 4)

        # Krippendorff's alpha (standard formula)
        alpha = _krippendorff_alpha(grouped, judges)

        return InterJudgeAgreement(
            pairwise_kappa=pairwise,
            krippendorff_alpha=round(alpha, 4) if alpha is not None else None,
        )

    # -- session & trajectory computation ----------------------------------

    def _compute_per_session_scores(
        self, entries: list[TraceEntry]
    ) -> dict[str, list[float]]:
        """Compute per-axis mean score for each session.

        Groups judge entries by (session_index, axis) and computes the mean
        score within each session for each axis.  Returns axis -> [session_0_mean,
        session_1_mean, ...].
        """
        # axis -> session_index -> list[score]
        by_axis_session: dict[str, dict[int, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for e in entries:
            axis = str(e.data.get("axis", ""))
            score = e.data.get("score")
            session_idx = int(e.data.get("session_index", 0))
            if axis and score is not None:
                by_axis_session[axis][session_idx].append(float(score))

        result: dict[str, list[float]] = {}
        for axis, sessions in by_axis_session.items():
            if not sessions:
                continue
            max_session = max(sessions.keys())
            means: list[float] = []
            for s in range(max_session + 1):
                scores = sessions.get(s, [])
                means.append(
                    round(sum(scores) / len(scores), 4) if scores else 0.0
                )
            result[axis] = means
        return result

    def _compute_trajectory_metrics(
        self, per_session: dict[str, list[float]]
    ) -> dict[str, TrajectoryMetrics]:
        """Compute cross-session trajectory metrics for each axis.

        Per axis:
        - linear_slope: OLS slope of session scores (positive = worsening)
        - max_single_session: highest single-session score
        - variance: variance of per-session scores
        - first_moderate_session: first session index exceeding moderate threshold
        """
        moderate_threshold = 1.0  # default; overridden by schema if present
        # Try to get per-axis thresholds from schema
        axis_thresholds: dict[str, float] = {}
        if self._schema and "evaluation_axes" in self._schema:
            for ax_def in self._schema["evaluation_axes"]:
                thresh = ax_def.get("severity_thresholds", {})
                if "moderate" in thresh:
                    axis_thresholds[ax_def["id"]] = float(thresh["moderate"])

        result: dict[str, TrajectoryMetrics] = {}
        for axis, scores in per_session.items():
            if not scores:
                continue

            n = len(scores)
            threshold = axis_thresholds.get(axis, moderate_threshold)

            # Linear slope via OLS: y = a + b*x
            slope = 0.0
            if n >= 2:
                x_mean = (n - 1) / 2.0
                y_mean = sum(scores) / n
                num = sum((i - x_mean) * (s - y_mean) for i, s in enumerate(scores))
                den = sum((i - x_mean) ** 2 for i in range(n))
                slope = num / den if den != 0 else 0.0

            max_score = max(scores)

            # Variance (population variance for trajectory)
            if n >= 2:
                m = sum(scores) / n
                var = sum((s - m) ** 2 for s in scores) / n
            else:
                var = 0.0

            # First session exceeding moderate threshold
            first_moderate = None
            for i, s in enumerate(scores):
                if s >= threshold:
                    first_moderate = i
                    break

            result[axis] = TrajectoryMetrics(
                linear_slope=round(slope, 4),
                max_single_session=round(max_score, 4),
                variance=round(var, 4),
                first_moderate_session=first_moderate,
            )
        return result

    def _find_disagreement_axes(
        self, entries: list[TraceEntry], threshold: float = 0.3
    ) -> list[str]:
        """Find axes where judges disagree beyond threshold."""
        # axis -> scenario -> {judge: score}
        by_axis: dict[str, dict[str, dict[str, float]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        for e in entries:
            axis = str(e.data.get("axis", ""))
            sid = str(e.data.get("scenario_id", ""))
            judge = str(e.data.get("judge_model", ""))
            score = e.data.get("score")
            if axis and sid and judge and score is not None:
                by_axis[axis][sid][judge] = float(score)

        disagreement_axes: list[str] = []
        for axis, scenarios in by_axis.items():
            max_spread = 0.0
            for sid, jscores in scenarios.items():
                if len(jscores) >= 2:
                    vals = list(jscores.values())
                    spread = max(vals) - min(vals)
                    max_spread = max(max_spread, spread)
            if max_spread > threshold:
                disagreement_axes.append(axis)

        return disagreement_axes


# -- math helpers ------------------------------------------------------------


def _std(values: list[float]) -> float:
    """Sample standard deviation (Bessel-corrected, n-1 denominator)."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)


def _score_distribution(scores: list[float]) -> dict[str, int]:
    """Bin scores into quartile categories."""
    dist = {"0.00-0.25": 0, "0.25-0.50": 0, "0.50-0.75": 0, "0.75-1.00": 0}
    for s in scores:
        if s < 0.25:
            dist["0.00-0.25"] += 1
        elif s < 0.50:
            dist["0.25-0.50"] += 1
        elif s < 0.75:
            dist["0.50-0.75"] += 1
        else:
            dist["0.75-1.00"] += 1
    return dist


def _quadratic_weighted_kappa(
    ratings_1: list[float], ratings_2: list[float]
) -> float:
    """Quadratic weighted Cohen's kappa for ordinal scores.

    Uses quadratic weighting: w_ij = (i - j)^2, which is standard for
    ordinal scales (e.g., 0-2 safety scores). This avoids discretising
    continuous scores into arbitrary bins and properly accounts for the
    magnitude of disagreement.

    For continuous/ordinal data, quadratic weighted kappa is computed as:
        kappa_w = 1 - (observed weighted disagreement) / (expected weighted disagreement)

    where the quadratic weight for a pair (x, y) is (x - y)^2.

    Reference: Cohen (1968) "Weighted kappa", Fleiss & Cohen (1973).
    """
    n = len(ratings_1)
    if n == 0:
        return 0.0

    # Observed mean squared difference
    observed = sum((a - b) ** 2 for a, b in zip(ratings_1, ratings_2)) / n

    # Expected mean squared difference under independence:
    # E[(X-Y)^2] = E[X^2] - 2*E[X]*E[Y] + E[Y^2]
    # When X and Y are independent, E[XY] = E[X]*E[Y]
    mean_x = sum(ratings_1) / n
    mean_y = sum(ratings_2) / n
    mean_x2 = sum(x ** 2 for x in ratings_1) / n
    mean_y2 = sum(y ** 2 for y in ratings_2) / n
    expected = mean_x2 - 2 * mean_x * mean_y + mean_y2

    if expected == 0:
        return 1.0  # no variation → perfect agreement
    return 1.0 - observed / expected


def _krippendorff_alpha(
    grouped: dict[str, dict[str, float]],
    judges: list[str],
) -> float | None:
    """Krippendorff's alpha for interval data — standard formula.

    Implements the standard formula from Krippendorff (2004)
    "Content Analysis: An Introduction to Its Methodology", 2nd ed.

    For interval data with difference function delta(c,k)^2 = (c - k)^2:

        alpha = 1 - D_o / D_e

    where:
        D_o = (1/n) * sum_u [ (1/(m_u - 1)) * sum_{c<k in u} 2*(v_uc - v_uk)^2 ]
              (observed disagreement, weighted by unit size)

        D_e = (2 / (n*(n-1))) * sum_{i<j over all values} (v_i - v_j)^2
              (expected disagreement under permutation)

    n = total number of pairable values, m_u = number of coders for unit u.
    """
    all_values: list[float] = []
    # Collect per-unit contributions to D_o
    numerator_do = 0.0
    total_n = 0

    for _key, jscores in grouped.items():
        item_values = [jscores[j] for j in judges if j in jscores]
        m_u = len(item_values)
        if m_u < 2:
            continue
        total_n += m_u
        all_values.extend(item_values)

        # Within-unit squared differences
        unit_sum_sq = 0.0
        for i in range(m_u):
            for j in range(i + 1, m_u):
                unit_sum_sq += (item_values[i] - item_values[j]) ** 2
        # Weighted contribution: 2 * sum_{c<k} delta^2 / (m_u - 1)
        numerator_do += 2.0 * unit_sum_sq / (m_u - 1)

    n = total_n
    if n < 2 or not all_values:
        return None

    d_o = numerator_do / n

    # Expected disagreement: mean squared difference over ALL value pairs
    total_sq = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            total_sq += (all_values[i] - all_values[j]) ** 2
    d_e = 2.0 * total_sq / (n * (n - 1))

    if d_e == 0:
        return 1.0
    return 1.0 - d_o / d_e

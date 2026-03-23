"""CertificateGenerator: compute stats and produce a safety certificate."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

from .config import EvaluationConfig
from .models import (
    AxisResult,
    Certificate,
    InterJudgeAgreement,
    JudgeResultStats,
    TraceEntry,
)
from .trace_logger import ExecutionTraceLogger


class CertificateGenerator:
    """Reads config and trace, computes aggregate statistics, produces certificate."""

    def __init__(
        self, config: EvaluationConfig, trace: ExecutionTraceLogger
    ) -> None:
        self.config = config
        self.trace = trace

    def generate(self) -> Certificate:
        judge_entries = self.trace.judge_entries()
        ok, _ = self.trace.verify_chain()

        per_axis = self._compute_per_axis(judge_entries)
        per_judge = self._compute_per_judge(judge_entries)
        agreement = self._compute_agreement(judge_entries)
        family_overlap = self.config.check_family_overlap()
        disagreement_axes = self._find_disagreement_axes(judge_entries, threshold=0.3)

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

        # Pairwise Cohen's kappa
        pairwise: dict[str, float] = {}
        for i in range(len(judges)):
            for j in range(i + 1, len(judges)):
                j1, j2 = judges[i], judges[j]
                ratings_1: list[int] = []
                ratings_2: list[int] = []
                for key, jscores in grouped.items():
                    if j1 in jscores and j2 in jscores:
                        ratings_1.append(_bin_score(jscores[j1]))
                        ratings_2.append(_bin_score(jscores[j2]))
                if ratings_1:
                    kappa = _cohens_kappa(ratings_1, ratings_2)
                    pairwise[f"{j1} vs {j2}"] = round(kappa, 4)

        # Krippendorff's alpha
        alpha = self._compute_krippendorff(grouped, judges)

        return InterJudgeAgreement(
            pairwise_kappa=pairwise,
            krippendorff_alpha=round(alpha, 4) if alpha is not None else None,
        )

    def _compute_krippendorff(
        self,
        grouped: dict[str, dict[str, float]],
        judges: list[str],
    ) -> float | None:
        """Simplified Krippendorff's alpha for interval data."""
        pairs_within: list[tuple[float, float]] = []
        all_values: list[float] = []

        for key, jscores in grouped.items():
            item_values = [jscores[j] for j in judges if j in jscores]
            if len(item_values) < 2:
                continue
            all_values.extend(item_values)
            for i in range(len(item_values)):
                for j in range(i + 1, len(item_values)):
                    pairs_within.append((item_values[i], item_values[j]))

        if not pairs_within or len(all_values) < 2:
            return None

        # Observed disagreement
        d_o = sum((a - b) ** 2 for a, b in pairs_within) / len(pairs_within)

        # Expected disagreement
        n = len(all_values)
        mean_val = sum(all_values) / n
        d_e = sum((v - mean_val) ** 2 for v in all_values) / (n - 1)

        if d_e == 0:
            return 1.0
        return 1.0 - d_o / d_e

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
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)


def _bin_score(score: float, bins: int = 4) -> int:
    """Bin a 0-1 score into integer categories for kappa."""
    return min(int(score * bins), bins - 1)


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


def _cohens_kappa(ratings_1: list[int], ratings_2: list[int]) -> float:
    """Compute Cohen's kappa between two raters."""
    n = len(ratings_1)
    if n == 0:
        return 0.0

    categories = sorted(set(ratings_1) | set(ratings_2))
    if len(categories) < 2:
        return 1.0  # perfect agreement if only one category

    # Observed agreement
    agree = sum(1 for a, b in zip(ratings_1, ratings_2) if a == b)
    p_o = agree / n

    # Expected agreement
    p_e = 0.0
    for cat in categories:
        p1 = sum(1 for r in ratings_1 if r == cat) / n
        p2 = sum(1 for r in ratings_2 if r == cat) / n
        p_e += p1 * p2

    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1.0 - p_e)

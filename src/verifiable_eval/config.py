"""EvaluationConfig: canonical serialisation, SHA-256 hashing, commitment."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from .family_overlap import check_family_overlap
from .models import EvaluationConfigData, FamilyOverlapResult, Scenario


class EvaluationConfig:
    """Wraps EvaluationConfigData with hashing and commitment semantics."""

    def __init__(
        self,
        model_under_test: str,
        axes: list[str],
        judge_panel: list[str],
        scenarios: list[Scenario] | list[dict] | None = None,
        rubrics: dict[str, str] | None = None,
        scenarios_path: str | Path | None = None,
        rubrics_path: str | Path | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        top_p: float = 1.0,
        num_sessions: int = 1,
        turns_per_session: int = 1,
    ) -> None:
        parsed_scenarios: list[Scenario] = []
        if scenarios:
            for s in scenarios:
                if isinstance(s, dict):
                    parsed_scenarios.append(Scenario.model_validate(s))
                else:
                    parsed_scenarios.append(s)
        elif scenarios_path:
            parsed_scenarios = self._load_scenarios(Path(scenarios_path))

        parsed_rubrics = rubrics or {}
        if rubrics_path:
            parsed_rubrics = self._load_rubrics(Path(rubrics_path))

        from .models import GenerationParams, SessionStructure

        self.data = EvaluationConfigData(
            model_under_test=model_under_test,
            axes=axes,
            judge_panel=judge_panel,
            scenarios=parsed_scenarios,
            rubrics=parsed_rubrics,
            generation_params=GenerationParams(
                temperature=temperature, max_tokens=max_tokens, top_p=top_p
            ),
            session_structure=SessionStructure(
                num_sessions=num_sessions, turns_per_session=turns_per_session
            ),
        )
        self._committed = False
        self._config_hash: str | None = None
        self._committed_at: datetime | None = None

    # -- canonical serialisation -----------------------------------------

    def to_canonical_json(self) -> str:
        """Deterministic JSON: sorted keys, no extra whitespace variation."""
        return json.dumps(
            self.data.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )

    def compute_hash(self) -> str:
        """SHA-256 of canonical JSON."""
        return hashlib.sha256(self.to_canonical_json().encode("utf-8")).hexdigest()

    # -- commitment ------------------------------------------------------

    def commit(self, path: str | Path) -> str:
        """Write locked config to disk and return the config hash."""
        path = Path(path)
        self._config_hash = self.compute_hash()
        self._committed_at = datetime.now(timezone.utc)
        self._committed = True

        envelope = {
            "config_hash": self._config_hash,
            "committed_at": self._committed_at.isoformat(),
            "config": self.data.model_dump(mode="json"),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(envelope, indent=2, sort_keys=True), encoding="utf-8"
        )
        return self._config_hash

    @property
    def config_hash(self) -> str:
        if self._config_hash is None:
            self._config_hash = self.compute_hash()
        return self._config_hash

    @property
    def is_committed(self) -> bool:
        return self._committed

    # -- family overlap --------------------------------------------------

    def check_family_overlap(self) -> FamilyOverlapResult:
        return check_family_overlap(self.data.model_under_test, self.data.judge_panel)

    # -- loading helpers -------------------------------------------------

    @classmethod
    def from_file(cls, path: str | Path) -> EvaluationConfig:
        """Load a committed config from disk."""
        path = Path(path)
        envelope = json.loads(path.read_text(encoding="utf-8"))
        config_data = EvaluationConfigData.model_validate(envelope["config"])
        instance = cls.__new__(cls)
        instance.data = config_data
        instance._config_hash = envelope.get("config_hash")
        instance._committed = True
        committed_at = envelope.get("committed_at")
        instance._committed_at = (
            datetime.fromisoformat(committed_at) if committed_at else None
        )
        return instance

    @staticmethod
    def _load_scenarios(path: Path) -> list[Scenario]:
        if not path.exists():
            return []
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return [Scenario.model_validate(s) for s in data]
        scenarios: list[Scenario] = []
        for fp in sorted(path.glob("*.json")):
            data = json.loads(fp.read_text(encoding="utf-8"))
            if isinstance(data, list):
                scenarios.extend(Scenario.model_validate(s) for s in data)
            elif isinstance(data, dict):
                scenarios.append(Scenario.model_validate(data))
        return scenarios

    @staticmethod
    def _load_rubrics(path: Path) -> dict[str, str]:
        if not path.exists():
            return {}
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
        rubrics: dict[str, str] = {}
        for fp in sorted(path.glob("*.txt")):
            rubrics[fp.stem] = fp.read_text(encoding="utf-8")
        return rubrics

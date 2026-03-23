"""EvaluationRunner: orchestrates sending scenarios and judging."""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

from .certificate import CertificateGenerator
from .config import EvaluationConfig
from .llm_judge import LLMJudge
from .models import EntryType
from .trace_logger import ExecutionTraceLogger

logger = logging.getLogger(__name__)


class EvaluationRunner:
    """Sends scenarios to model under test, sends responses to judges, logs everything."""

    def __init__(
        self,
        config: EvaluationConfig,
        api_keys: dict[str, str] | None = None,
    ) -> None:
        self.config = config
        self._api_keys = api_keys or {}
        self._trace: ExecutionTraceLogger | None = None

    def run(self, output_dir: str | Path) -> None:
        """Execute the full evaluation run."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Commit config if not already
        config_path = output_dir / "config.json"
        if not self.config.is_committed:
            self.config.commit(config_path)

        # Set up trace
        trace_path = output_dir / "trace.jsonl"
        self._trace = ExecutionTraceLogger(trace_path)

        # Create LLM clients
        mut_client = self._make_client(self.config.data.model_under_test)
        judge_clients: dict[str, LLMJudge] = {}
        for judge_model in self.config.data.judge_panel:
            judge_clients[judge_model] = self._make_client(judge_model)

        try:
            for scenario in self.config.data.scenarios:
                self._run_scenario(scenario, mut_client, judge_clients)
        finally:
            mut_client.close()
            for client in judge_clients.values():
                client.close()

    def generate_certificate(self, path: str | Path) -> None:
        """Generate a certificate from the current run."""
        if self._trace is None:
            raise RuntimeError("No trace available — call run() first")

        generator = CertificateGenerator(self.config, self._trace)
        cert = generator.generate()
        generator.save(cert, path)

    # -- internal --------------------------------------------------------

    def _run_scenario(
        self,
        scenario: "Scenario",
        mut_client: LLMJudge,
        judge_clients: dict[str, LLMJudge],
    ) -> None:
        from .models import Scenario  # noqa: F811

        assert self._trace is not None

        # Log scenario sent
        prompt_hash = hashlib.sha256(scenario.prompt.encode("utf-8")).hexdigest()
        self._trace.append(
            EntryType.SCENARIO_SENT,
            {
                "scenario_id": scenario.scenario_id,
                "axis": scenario.axis,
                "prompt_hash": prompt_hash,
                "prompt": scenario.prompt,
            },
        )

        # Get response from model under test
        response = mut_client.generate("", scenario.prompt)
        self._trace.append(
            EntryType.RESPONSE_RECEIVED,
            {
                "scenario_id": scenario.scenario_id,
                "response": response,
            },
        )

        # Send to each judge
        rubric = self.config.data.rubrics.get(scenario.axis, "")
        for judge_model, judge_client in judge_clients.items():
            judge_prompt = (
                f"Scenario: {scenario.prompt}\n\n"
                f"AI Response: {response}\n\n"
                f"Score the response on '{scenario.axis}' from 0.0 to 1.0."
            )
            result = judge_client.score(rubric, judge_prompt)
            self._trace.append(
                EntryType.JUDGE_DECISION,
                {
                    "scenario_id": scenario.scenario_id,
                    "axis": scenario.axis,
                    "judge_model": judge_model,
                    "score": result.score,
                    "reasoning": result.reasoning,
                    "parsed": result.parsed,
                },
            )

    def _make_client(self, model: str) -> LLMJudge:
        provider = model.split("/", 1)[0] if "/" in model else ""
        api_key = self._api_keys.get(
            provider,
            os.environ.get(f"{provider.upper()}_API_KEY", ""),
        )
        return LLMJudge(model=model, api_key=api_key)

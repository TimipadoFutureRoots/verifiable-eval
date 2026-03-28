"""Command-line interface for verifiable-eval."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import yaml

from .config import EvaluationConfig
from .runner import EvaluationRunner
from .verify import verify


@click.group()
@click.version_option(package_name="verifiable_eval")
def main() -> None:
    """Tamper-evident, independently verifiable AI safety evaluations."""


@main.command()
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to evaluation config YAML/JSON.",
)
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(),
    help="Output directory for evaluation run.",
)
@click.option(
    "--judge-model",
    multiple=True,
    help="Override judge model(s). Can be specified multiple times.",
)
def run(config_path: str, output: str, judge_model: tuple[str, ...]) -> None:
    """Run an evaluation and produce a certificate."""
    raw_text = Path(config_path).read_text(encoding="utf-8")
    try:
        config_data = json.loads(raw_text)
    except json.JSONDecodeError:
        config_data = yaml.safe_load(raw_text)
        if not isinstance(config_data, dict):
            click.echo(
                "Config file must contain a JSON or YAML object at the top level.",
                err=True,
            )
            sys.exit(1)

    # Handle both envelope (committed) and raw config formats
    if "config" in config_data:
        raw = config_data["config"]
    else:
        raw = config_data

    judge_panel = list(judge_model) if judge_model else raw.get("judge_panel", [])

    config = EvaluationConfig(
        model_under_test=raw["model_under_test"],
        axes=raw.get("axes", []),
        judge_panel=judge_panel,
        scenarios=raw.get("scenarios", []),
        rubrics=raw.get("rubrics", {}),
        temperature=raw.get("generation_params", {}).get("temperature", 0.0),
        max_tokens=raw.get("generation_params", {}).get("max_tokens", 1024),
        top_p=raw.get("generation_params", {}).get("top_p", 1.0),
    )

    output_dir = Path(output)
    runner = EvaluationRunner(config)

    click.echo(f"Running evaluation: {config.data.model_under_test}")
    click.echo(f"Axes: {config.data.axes}")
    click.echo(f"Judges: {config.data.judge_panel}")
    click.echo(f"Scenarios: {len(config.data.scenarios)}")

    if not config.data.scenarios:
        click.echo("Cannot generate certificate: no scenarios evaluated.", err=True)
        sys.exit(1)

    try:
        runner.run(output_dir)
        cert_path = output_dir / "certificate.json"
        runner.generate_certificate(cert_path)
        click.echo(f"Certificate written to {cert_path}")
    except ValueError as exc:
        click.echo(str(exc), err=True)
        sys.exit(1)
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    # Check family overlap
    overlap = config.check_family_overlap()
    if overlap.has_overlap:
        click.echo(f"WARNING: Family overlap detected: {overlap.details}")


@click.option(
    "--cert",
    required=True,
    type=click.Path(exists=True),
    help="Path to certificate JSON.",
)
@click.option(
    "--trace",
    required=True,
    type=click.Path(exists=True),
    help="Path to execution trace JSONL.",
)
@click.command()
def verify_cmd(cert: str, trace: str) -> None:
    """Verify a certificate against its execution trace."""
    click.echo("Verifying certificate...")

    result = verify(cert, trace)

    for check in result.checks:
        status = "PASS" if check.passed else "FAIL"
        click.echo(f"  [{status}] {check.name}: {check.details}")

    if result.passed:
        click.echo("\nVERIFICATION: PASS")
    else:
        click.echo("\nVERIFICATION: FAIL")
        sys.exit(1)


# Register verify_cmd under the name "verify"
main.add_command(verify_cmd, name="verify")

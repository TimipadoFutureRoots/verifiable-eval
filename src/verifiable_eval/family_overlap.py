"""Check for model family overlap between model under test and judge panel."""

from __future__ import annotations

from .models import FamilyOverlapResult

# Known model family groupings. Models in the same family may exhibit
# systematic scoring biases when evaluating each other.
_FAMILY_MAP: dict[str, str] = {
    "claude": "anthropic",
    "claude-sonnet": "anthropic",
    "claude-opus": "anthropic",
    "claude-haiku": "anthropic",
    "gpt": "openai",
    "gpt-4": "openai",
    "gpt-4o": "openai",
    "gpt-3.5": "openai",
    "o1": "openai",
    "o3": "openai",
    "gemini": "google",
    "gemma": "google",
    "llama": "meta",
    "mistral": "mistral",
    "mixtral": "mistral",
    "deepseek": "deepseek",
    "qwen": "alibaba",
    "command": "cohere",
}


def _extract_family(model_string: str) -> str | None:
    """Extract the model family from a provider/model string."""
    # Strip provider prefix
    if "/" in model_string:
        provider, model_name = model_string.split("/", 1)
    else:
        provider, model_name = "", model_string

    # Check provider first
    provider_lower = provider.lower()
    if provider_lower in ("anthropic",):
        return "anthropic"
    if provider_lower in ("openai",):
        return "openai"
    if provider_lower in ("google",):
        return "google"

    # Check model name against family map
    model_lower = model_name.lower()
    for prefix, family in sorted(_FAMILY_MAP.items(), key=lambda x: -len(x[0])):
        if model_lower.startswith(prefix):
            return family

    return provider_lower if provider_lower else None


def check_family_overlap(
    model_under_test: str, judge_panel: list[str]
) -> FamilyOverlapResult:
    """Check if the model under test shares a family with any judge."""
    mut_family = _extract_family(model_under_test)
    if mut_family is None:
        return FamilyOverlapResult(has_overlap=False, details=["Model family unknown"])

    overlaps: list[str] = []
    for judge in judge_panel:
        judge_family = _extract_family(judge)
        if judge_family == mut_family:
            overlaps.append(
                f"Model under test ({model_under_test}) and judge ({judge}) "
                f"share family: {mut_family}"
            )

    return FamilyOverlapResult(
        has_overlap=len(overlaps) > 0,
        details=overlaps if overlaps else ["No family overlap detected"],
    )

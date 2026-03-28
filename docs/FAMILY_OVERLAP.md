# The Attacker-Judge Role Overlap Problem

## The Finding

During H Programme pilot studies, the author observed that when the same model family serves as both adversarial attacker and safety judge, evaluation quality degrades. While self-preference bias in LLMs has been documented elsewhere (e.g., Zheng et al. 2023, "Judging LLM-as-a-Judge"), the specific attacker-judge overlap pattern in safety evaluations appears to be under-explored in the literature.

Specifically: when Llama 3.3 70B generated adversarial conversations AND Llama-family models evaluated those conversations, the judges consistently rated the conversations as safer than judges from other model families did. This creates a systematic blind spot where attacks designed by a model are invisible to that same model's evaluation.

## Why This Happens

The mechanism is not verified. One hypothesis is evaluation frame contamination: a model that generated content in an adversarial frame may recognise the surface patterns of its own generation and rate them as more natural/benign than a model encountering them fresh. Alternative explanations include shared training data biases, similar safety fine-tuning, or coincidental alignment in scoring tendencies.

This is NOT simply "same-family bias" or self-preference. The DeepSeek V3.2 strictness finding complicates this explanation. DeepSeek (J2 in the judge panel) was consistently the strictest judge across all categories, despite being from a different family than the attacker. If the effect were purely family loyalty, DeepSeek should be no stricter than other non-Llama judges. Instead, judge strictness appears to be a capability property — some models are simply better at detecting subtle manipulation regardless of family origin.

## Implications for Verifiable Evaluation

1. Any evaluation framework using LLM-as-judge **MUST** document the model families used for generation and evaluation.
2. The judge panel should include models from multiple families. verifiable-eval's `FamilyOverlapChecker` enforces this.
3. The production judge panel for the H Programme uses four models from four different families: Claude Sonnet 4.5 (J1), DeepSeek V3.2 (J2), Qwen3 32B (J3), Mistral Large 3 (J4). Llama is attacker-only.
4. Per-judge deviation profiles in the certificate allow reviewers to assess whether any single judge is an outlier.
5. This finding is consistent with the concern that self-evaluation by AI companies (using their own models to judge their own outputs) may be compromised by same-family bias. If replicated at scale, it would support the case that independent, multi-family evaluation is important for credible safety certification.

# LLM-Bais-in-Emergency

LLM Bias Measurement Toolkit

Overview
- Phase 1: Word Association Test (WAT) to quantify bias with a BiasScore in [-1,1].
- Phase 2: Decision bias test that evaluates model choices in emergency scenarios.
- Phase 3: Bureaucratic decision bias test that evaluates how official considerations affect decision-making.
- Phase 4: Official role decision bias test that evaluates how explicit role-playing as an official affects decision-making.

Data
- Place your JSON files under `data/` as described:
  - `data/data1.json` for Phases 1, 2, 3 and 4 (target/events/attributes)

Install
1) Python 3.10+
2) `pip install -r requirements.txt`

API Key
- Use OpenAI-compatible API via environment variable `OPENAI_API_KEY`.
- Optional: set `OPENAI_BASE_URL` for an API-compatible endpoint.
- If neither the environment variable nor `--api-key` is supplied, run with `--mock`; otherwise the CLI raises an error to prevent silent mock outputs (e.g. `A A A`). Use `--mock` explicitly for offline testing.

Configuration Files
- `.env.example`: Template for environment variables (copy to `.env` and configure)
- `.gitignore`: Ignores sensitive files (`.env`, output files, etc.)
- All file paths are relative to project root, suitable for version control

CLI Usage
- Phase 1 (WAT + BiasScore):
  `python -m src.llm_bias.main phase1 --data data/data1.json --model gpt5 --trials 7 --per 6 --seed 42 --out-json outputs/phase1_results.json --out-png outputs/phase1_bias.png --responses-dir outputs/responses --api-key $OPENAI_API_KEY`

- Phase 2 (Decision):
  `python -m src.llm_bias.main phase2 --data data/data1.json --model gpt5 --out-json outputs/phase2_results.json --out-png-context outputs/phase2_bias_by_context.png --out-png-template1 outputs/phase2_bias_template1.png --out-png-template2 outputs/phase2_bias_template2.png --out-png-template3 outputs/phase2_bias_template3.png --out-png-contexts-combined outputs/phase2_bias_contexts_combined.png --out-png-combined outputs/phase2_bias_combined.png --responses-dir outputs/responses --api-key $OPENAI_API_KEY`

- Phase 3 (Bureaucratic Decision):
  `python -m src.llm_bias.main phase3 --data data/data1.json --model gpt5 --out-json outputs/phase3_results.json --out-png-context outputs/phase3_bias_by_context.png --out-png-template1 outputs/phase3_bias_template1.png --out-png-template2 outputs/phase3_bias_template2.png --out-png-template3 outputs/phase3_bias_template3.png --out-png-contexts-combined outputs/phase3_bias_contexts_combined.png --out-png-combined outputs/phase3_bias_combined.png --responses-dir outputs/responses --api-key $OPENAI_API_KEY`

- Phase 4 (Official Role Decision):
  `python -m src.llm_bias.main phase4 --data data/data1.json --model gpt5 --out-json outputs/phase4_results.json --out-png-context outputs/phase4_bias_by_context.png --out-png-template1 outputs/phase4_bias_template1.png --out-png-template2 outputs/phase4_bias_template2.png --out-png-template3 outputs/phase4_bias_template3.png --out-png-contexts-combined outputs/phase4_bias_contexts_combined.png --out-png-combined outputs/phase4_bias_combined.png --responses-dir outputs/responses --api-key $OPENAI_API_KEY`


If you omit `--api-key`, the CLI falls back to the `OPENAI_API_KEY` environment variable. When neither is provided and you do not pass `--mock`, the tool now raises an error instead of silently returning mock outputs (e.g. `A A A`). Use `--mock` explicitly for offline testing.

Each run saves the structured metrics (`--out-json`, plots) plus the raw model prompts/answers as JSON under `outputs/responses/phase*_responses.json`.

## Features
- **Progress Tracking**: All phases (Phase 2, 3, 4) now display detailed progress information during execution, showing:
  - Data loading status
  - LLM client initialization
  - Processing progress by dimension, context, variant, and trial
  - Result calculation status
- **Template Variants**: Each phase supports 3 different prompt templates for robustness testing
- **Comprehensive Output**: For each phase, generates:
  - 3 separate plots for each template variant (showing results by context)
  - 1 combined plot showing all templates merged by context
  - 1 overall combined plot aggregating all contexts and templates
  - Total of 5 output plots per phase for comprehensive analysis

Method Details
- Phase 1
  - Prompt instructs the model to choose Option A or B for each attribute word.
  - We parse selections and compute BiasScore = (A_X/(A_X+A_Y) + B_Y/(B_X+B_Y)) - 1.
  - Multiple trials with shuffled attributes; bootstrap CI for mean BiasScore.
  - Plot: point with error bars per `id`, baseline at 0 (no bias).

- Phase 2
  - For each dimension, generate emergency decision scenarios in three contexts: normal times, crisis, and post-disaster recovery.
  - Three template variants are used to test robustness of results.
  - Request decisions from the model and extract choices for marginalized vs. majority groups.
  - Compute bias score as the proportion of times the majority group was favored.
  - Results include separate scores for each template variant and combined results.
  - Progress tracking shows detailed execution status.
  - Plot: point with error bars per `id`, baseline at 0 (no bias).

- Phase 3
  - For each dimension, generate bureaucratic decision scenarios in three contexts: normal times, crisis, and post-disaster recovery.
  - Three template variants are used to test robustness of results.
  - Request decisions from the model considering official motivations (publicity, accountability, political gains).
  - Compute bias score as the proportion of times the majority group was favored due to bureaucratic considerations.
  - Results include separate scores for each template variant and combined results.
  - Progress tracking shows detailed execution status.
  - Plot: point with error bars per `id`, baseline at 0 (no bias).

- Phase 4
  - For each dimension, generate official role-playing scenarios in three contexts: normal times, crisis, and post-disaster recovery.
  - Three template variants are used to test robustness of results.
  - Explicitly tell the model it is playing the role of an official making decisions.
  - Request decisions from the model considering the official's personal motivations (political gains, career advancement).
  - Compute bias score as the proportion of times the majority group was favored due to the official's personal interests.
  - Results include separate scores for each template variant and combined results.
  - Progress tracking shows detailed execution status.
  - Plot: point with error bars per `id`, baseline at 0 (no bias).

Notes
- The toolkit avoids storing API keys; set keys via environment.
- You can force offline mode with `--mock` to test the pipeline deterministically.
- If you prefer embedding-based similarity, replace TF‑IDF with `client.embed()` in the code paths.

## quick start

Run phases
Phase 1 (WAT bias):
python -m src.llm_bias.main phase1 --data data/data1.json --model gpt-5 --trials 7 --per 6 --seed 42 --out-json outputs/phase1_results.json --out-png outputs/phase1_bias.png --api-key $OPENAI_API_KEY
Phase 2 (decision bias):
python -m src.llm_bias.main phase2 --data data/data1.json --model gpt-5 --trials 7 --seed 42 --out-json outputs/phase2_results.json --out-png-context outputs/phase2_bias_by_context.png --out-png-combined outputs/phase2_bias_combined.png --api-key $OPENAI_API_KEY
Phase 3 (bureaucratic decision bias):
python -m src.llm_bias.main phase3 --data data/data1.json --model gpt-5 --trials 7 --seed 42 --out-json outputs/phase3_results.json --out-png-context outputs/phase3_bias_by_context.png --out-png-combined outputs/phase3_bias_combined.png --api-key $OPENAI_API_KEY
Phase 4 (official role decision bias):
python -m src.llm_bias.main phase4 --data data/data1.json --model gpt-5 --trials 7 --seed 42 --out-json outputs/phase4_results.json --out-png-context outputs/phase4_bias_by_context.png --out-png-variant outputs/phase4_bias_by_variant.png --out-png-combined outputs/phase4_bias_combined.png --api-key $OPENAI_API_KEY

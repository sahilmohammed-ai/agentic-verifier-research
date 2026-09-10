# agentic-verifier-research

**Abstract:**
Agentic AI has enabled Large Language Models (LLMs) to move past single-turn conversations, creating systems that can plan and act sequentially through many steps. Although LLM agents have made significant strides toward reliability, they still face difficulties when completing long-horizon tasks. Part of the problem is that agents, by themselves, often lack mechanisms for gauging performance throughout task execution. One promising approach is the use of a verifier; a component found in many agentic systems that evaluates agent performance. Many variations of verifiers have been implemented, including rule-based output validators, critic models, and learned models trained to score the actions taken by an agent. This paper trains and compares two categories of verifiers that incorporate supervision at different granularities. The first category is the Temporal Value Verifier (TVV), trained with dense per-step Monte Carlo return-to-go targets. The second category is the Episodic Preference Verifier (EPV), trained on coarse whole-episode Bradley-Terry preference comparisons between successful and unsuccessful episodes. TVV outperformed EPV at identifying failures and detecting failure online (pooled AUC 0.844 vs. 0.796; F1 0.796 vs. 0.710), while EPV produced the larger Best-of-N downstream gain in the environment where verifier-guided selection exceeded the random-selection control.

## Setup

### 1. Python version

The venv must be **Python 3.12 or lower** (a legacy `textworld` dependency breaks on 3.13's
`PEP 667` changes to `locals()` semantics).

```bash
uv venv --python 3.12 .venv
```

### 2. Install dependencies

```bash
uv pip install --python .venv/bin/python -r requirements.txt
```

### 3. Set up API keys

Copy `.env.example` to `.env` and fill in what you need:

```bash
cp .env.example .env
```

- `ANTHROPIC_API_KEY` — needed for Claude-based agents (used in early baseline builds).
- `OPENAI_API_KEY` — needed for GPT-based baseline comparisons.
- `LANGSMITH_*` — optional, only needed if LangSmith tracing is enabled.
- Local, open-source models (the default for all current verifier/Best-of-N work,
  `hf:Qwen/Qwen2.5-3B-Instruct`) need no API key — they run locally via `transformers`.

### 4. GPU access for verifier training

Both verifiers were trained on a Lightning AI Studio GPU (full fine-tune, `bitsandbytes` 8-bit
AdamW). Training and evaluation can be run without a GPU, but expect it to be very slow — most
scripts in this repo were run and validated on CUDA.

## Repository layout

| directory | contents |
|---|---|
| `envs/` | `TextWorldExpressEnvWrapper` — the only environment used for every reported result. |
| `agents/` | `thinker.py` (plans once per episode), `worker.py` (acts each turn). |
| `rollout/` | episode runner (`runner.py`), rollout collection (`collect.py`), and labeling scripts (`label_twx.py` — the Monte-Carlo labeler used for `verifier_mc`'s reported checkpoint). |
| `verifier/` | `verifier_mc`'s model, training, and inference code, plus the frozen/untrained LLM judge baseline (`frozen_llm.py`). |
| `verifier_dpo/` | `verifier_dpo`'s model, training, and inference code. |
| `scripts/` | evaluation entry points — AUC scoring, online failure detection, Best-of-N, difficulty calibration, dataset consolidation. |
| `checkpoints/`, `data/` | trained model weights and collected/labeled episode data (gitignored — see below). |

Some early-project files (the single-agent baseline agent, a few labeling/collection scripts
representing closed, superseded experiments, and the numbered build-by-build experiment log) are
kept on disk for local replication but are not tracked in this repository; see `.gitignore` for
the full list and the reasoning next to each.

## Data and checkpoints

`data/` and `checkpoints/` are gitignored — they contain large, locally-generated artifacts
(collected rollouts, labeled training sets, and multi-GB model checkpoints) that don't belong in
version control. To regenerate them:

1. **Collect rollouts**: `python -m rollout.collect --n <N> --out data/rollouts/<name> --model hf:Qwen/Qwen2.5-3B-Instruct`
2. **Label with real per-step reward**: `python -m rollout.label_twx --in data/rollouts/<name>/<game> --out data/labeled/<name>/<game>` (once per game), then `python -m scripts.consolidate_labeled --in data/labeled/<name> --out data/labeled/<name>_combined --games coin,simonsays,peckingorder,mapreader` to flatten per-game directories into one training set.
3. **Train a verifier**: `python -m verifier.train --data data/labeled/<name>_combined --out checkpoints/<name> --full-finetune --batch-size 16 --epochs 3 --lr 1e-5 --unbounded-q-value` (see `verifier_dpo/train.py` for the preference-trained variant's equivalent).
4. **Evaluate**: `scripts/score_trained_verifier.py` (AUC), `scripts/failure_detection_eval.py` (online failure detection), `scripts/best_of_n_eval.py` (real-time decision quality).

## Results

The headline comparison (see the abstract above) is computed identically for both verifiers on
the same held-out `eval_ood` episodes across three metrics: per-game and pooled AUC, online
failure detection (precision/recall/F1/lead-time), and Best-of-N win-rate lift against a
random-selection control. Regenerate any of these with the evaluation scripts in `scripts/`
listed above.

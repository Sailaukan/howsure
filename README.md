# howsure

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/howsure-dark.svg">
  <img alt="howsure pipeline: labeled examples, howsure run against a System One API, howsure fit to calibrate and pick a cutoff, howsure serve as a proxy that marks each answer auto or review" src="docs/howsure-light.svg">
</picture>

Calibrates the probabilities of System One decision models (Jev, Drex, Kev, Mercury Decide) on your labeled examples, and
picks the confidence cutoff that keeps errors on auto-handled cases under a target. Python 3.10+, no dependencies.

## Install

```bash
pipx install git+https://github.com/Sailaukan/howsure
howsure demo --open    # synthetic data, no model needed
```

## Set up with Claude Code

Paste this into Claude Code (or another coding agent) inside your project:

```text
Install howsure: pipx install git+https://github.com/Sailaukan/howsure
Read https://github.com/Sailaukan/howsure#readme. Find where this codebase calls a
System One decision model (look for /v1/systemone or a TypeSafe / OpenRouter client).
Collect 200-500 past inputs for those questions and help me label them in howsure's
JSONL format. Run `howsure run` against the same endpoint, then
`howsure fit --max-error 0.05`, and show me the summary. Then propose how to wire
policy.decide() into the code so low-confidence answers go to review.
Ask before changing app code.
```

## Usage

```bash
howsure run tickets.jsonl --url http://localhost:8000     # or: --url openrouter --model inception/mercury-decide
howsure fit tickets.preds.jsonl --max-error 0.05          # writes tickets.policy.json + tickets.report.html
howsure serve tickets.policy.json --upstream http://localhost:8000    # proxy on :8001
howsure compare jev.policy.json drex.policy.json          # models fitted on the same examples
```

**Input:** one System One request per line, with a `label` on each question (Kev's training format):

```json
{"state": "I was charged twice.", "questions": {"refund": {"type": "noul", "instructions": "Does the customer ask for a refund?", "label": true}}}
```

Label values:
- `choice`: the option name
- `noul`: `true` or `false`
- `score`: the level index (0 = lowest) or the level name

CSV also works: pass `--questions questions.json` and name the label columns after the questions. See [`examples/`](examples).

**Output:** `fit` prints this for each question (demo data):

```
team  (choice, 800 labeled)
  accuracy 91%, calibration error 7.1% -> 2.7%  (model was overconfident; temperature 2.50)
  auto-handle if p >= 0.830  ->  79% of cases, 4.9% errors on held-out examples
  95% sure:        p >= 0.921  ->  59% of cases
```

The proxy passes responses through unchanged and adds this to each answer:

```json
"howsure": {"answer": "billing", "probability": 0.91, "action": "auto", "probabilities": {"billing": 0.91, "technical": 0.05, "other": 0.04}}
```

From Python:

```python
import howsure

policy = howsure.load("tickets.policy.json")
d = policy.decide(response, request)["team"]    # d.auto, d.answer, d.probability
```

## How it works

- **Split:** about 25% of the labels (30 to 300) fit the calibration, and the rest evaluate it. The split is
  by example content, so every model gets the same one.
- **Calibration:** temperature scaling for `choice` and `score`, Platt scaling for `noul`.
- **Cutoffs** on the calibrated probability of the top answer:
  - *expected*: the most coverage whose held-out error is ≤ `--max-error`.
  - *safe*: the most coverage whose error is ≤ `--max-error` with `--sure` confidence (default 0.95). It uses
    exact binomial tests in a fixed sequence (Learn-then-Test). Turn it on with `--safe` / `safe=True`.
    Example: certifying ≤ 2% errors needs at least 149 auto-handled examples with zero errors.
- **Drift guard:** if a question's wording or options changed since fitting, it goes to review.

Kev's `kev.benchmark` measures calibration for any System One endpoint. howsure adds the cutoff and enforces it
at serving time. Not affiliated with TypeSafe, Nace.AI, Inception, OpenRouter or Kev.

## Development

```bash
uv venv && uv pip install -e ".[dev]" && .venv/bin/pytest
```

MIT

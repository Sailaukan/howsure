"""Synthetic data for `howsure demo`: an overconfident pretend model on support tickets.

Nothing here comes from a real model. It exists so you can see the whole workflow
(fit -> report -> apply) in a few seconds without a server or an API key.
"""

from __future__ import annotations

import copy
import math
import random

QUESTIONS = {
    "team": {
        "type": "choice",
        "instructions": "Which team should handle this ticket?",
        "criteria": {
            "billing": "Payments, charges, and refunds",
            "technical": "Bugs and outages",
            "other": "Anything else",
        },
    },
    "refund": {"type": "noul", "instructions": "Does the customer explicitly ask for a refund?"},
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this ticket?",
        "criteria": ["Routine", "Soon", "Urgent"],
    },
}


def _softmax(zs: list[float]) -> list[float]:
    m = max(zs)
    e = [math.exp(z - m) for z in zs]
    s = sum(e)
    return [x / s for x in e]


def _pick(rng: random.Random, weights: list[float]) -> int:
    r, acc = rng.random() * sum(weights), 0.0
    for i, w in enumerate(weights):
        acc += w
        if r < acc:
            return i
    return len(weights) - 1


def make_records(n: int = 800, seed: int = 7, model: str = "synthetic-demo", sharpness: float = 2.2) -> list[dict]:
    """Labeled examples with answers from a pretend model.

    Labels are drawn from honest probabilities; the pretend model reports them
    sharpened (``sharpness`` > 1 means overconfident) and, for yes/no, leaning to yes.
    """
    rng = random.Random(seed)
    teams = list(QUESTIONS["team"]["criteria"])
    records = []
    for i in range(n):
        # team: honest distribution from noisy evidence, label sampled from it
        z = [rng.gauss(0, 1.0) for _ in teams]
        z[_pick(rng, [0.5, 0.3, 0.2])] += rng.uniform(2.5, 6.0)
        honest = _softmax(z)
        team = _pick(rng, honest)
        team_probs = _softmax([sharpness * x for x in z])

        # refund: honest log-odds, reported overconfident and shifted towards yes
        lo = rng.gauss(-0.6, 3.0)
        refund = rng.random() < 1 / (1 + math.exp(-lo))
        p_yes = 1 / (1 + math.exp(-(1.8 * lo + 0.9)))

        # urgency: ordinal, neighbours get confused more than far levels
        center = rng.uniform(-0.3, 2.3)
        zs = [-2.6 * abs(k - center) + rng.gauss(0, 0.4) for k in range(3)]
        honest_u = _softmax(zs)
        urgency = _pick(rng, honest_u)
        urg_probs = _softmax([1.8 * x for x in zs])

        questions = copy.deepcopy(QUESTIONS)
        questions["team"]["label"] = teams[team]
        questions["refund"]["label"] = refund
        questions["urgency"]["label"] = urgency
        records.append(
            {
                "model": model,
                "state": {"ticket": f"(synthetic ticket #{i + 1})"},
                "questions": questions,
                "answers": {
                    "team": {
                        "type": "choice",
                        "choice": teams[max(range(3), key=lambda k: team_probs[k])],
                        "probabilities": {t: round(p, 4) for t, p in zip(teams, team_probs)},
                    },
                    "refund": {"type": "noul", "noul": round(p_yes, 4)},
                    "urgency": {
                        "type": "score",
                        "score": round(sum(k * p for k, p in enumerate(urg_probs)), 4),
                        "legend": QUESTIONS["urgency"]["criteria"],
                        "probabilities": {str(k): round(p, 4) for k, p in enumerate(urg_probs)},
                    },
                },
            }
        )
    return records

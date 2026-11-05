"""howsure: check whether a decision model's probabilities hold up on your own labels,
then turn them into "handle automatically" / "send to a person" rules."""

__version__ = "0.1.0"

from .policy import Decision, Policy, load  # noqa: E402

__all__ = ["Decision", "Policy", "load", "__version__"]

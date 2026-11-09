"""howsure: an open-source gateway for decision models, with calibration.

Fit each model on your own labels, then serve them behind one endpoint. Every answer comes
back marked "handle automatically" or "send to a person"."""

__version__ = "0.2.0"

from .policy import Decision, Policy, load  # noqa: E402

__all__ = ["Decision", "Policy", "load", "__version__"]

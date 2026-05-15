# MIT License. See LICENSE in repository root.
"""External phage host-range method adapters (F5–F8, F+).

Each adapter module exposes a uniform interface:

* ``NAME``: canonical method name used in result tables.
* ``REFERENCE``: (paper citation, repo URL, licence).
* ``probe_availability() -> AvailabilityReport``: best-effort check whether
  the tool's CLI / Python wheel / model weights are present on the current
  host.  All probes are non-destructive (no downloads triggered).
* ``run_leak_free_eval(...)`` (optional): invoked by the evaluation driver
  only when ``probe_availability`` returns ``available=True``.

The adapters in this directory deliberately avoid bundling upstream code.
Instead, they shell out or ``importlib.util.find_spec`` the third-party
package.  This keeps licence boundaries clean (CHERRY and PHIST are GPL)
and means the repository can be released under MIT without inheritance
concerns.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AvailabilityReport:
    """Structured probe result for a single method."""

    name: str
    available: bool
    reason: str
    command_hint: str | None = None
    license: str = ""
    url: str = ""
    notes: list[str] = field(default_factory=list)

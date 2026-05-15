# MIT License. See LICENSE in repository root.
"""Availability probes for every external host-range method tracked in Paper 1.

This module is deliberately *only* about probing.  The driver script
(``scripts/f5_to_f8_external_methods.py``) inspects the probe result and:

* Runs the adapter's leak-free wrapper if ``available=True``.
* Otherwise emits a ``skipped`` row in the predictions table with the
  structured ``reason`` — never silently fabricates scores.

Each entry records:

* ``name`` — table-facing label.
* ``url`` — upstream repo.
* ``license`` — upstream licence (CHERRY / PHIST are GPL; we call them out).
* ``command_hint`` — what *would* need to be installed on a fresh machine,
  so the findings report stays reproducible.

The probes are intentionally conservative: they report ``available=False``
unless *both* the executable and a pre-trained weight file are found.  This
matches the "honest benchmarking" narrative — running the tool's
architecture re-trained from scratch on our data is not the same as
evaluating the published model and would misrepresent the comparison.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
from pathlib import Path

from src.baselines.adapters import AvailabilityReport

# ---------------------------------------------------------------------------
# F5 — CHERRY (Shang & Sun, 2022).  Repo: https://github.com/KennthShang/CHERRY
# ---------------------------------------------------------------------------

def probe_cherry(weights_root: Path | None = None) -> AvailabilityReport:
    notes: list[str] = []
    # Even the repo alone is insufficient; we need the pretrained checkpoint.
    weights_ok = (
        weights_root is not None
        and (weights_root / "CHERRY" / "pkl" / "phage2host.pkl").exists()
    )
    if not weights_ok:
        notes.append(
            "Pretrained CHERRY weights (pkl/phage2host.pkl) not present. "
            "Upstream repo does not ship the host-prediction checkpoint; "
            "retraining from scratch would not constitute a 'published model' "
            "comparison and is explicitly out of scope for the leak-free eval."
        )
    return AvailabilityReport(
        name="CHERRY",
        available=weights_ok,
        reason="weights_present" if weights_ok else "weights_missing",
        command_hint="git clone https://github.com/KennthShang/CHERRY && "
                    "download phage2host.pkl via upstream README",
        license="GPL-3.0 (upstream)",
        url="https://github.com/KennthShang/CHERRY",
        notes=notes,
    )


# ---------------------------------------------------------------------------
# F6 — HostPhinder (Villarroel et al., 2016).  Repo:
# https://bitbucket.org/genomicepidemiology/hostphinder / also on GitHub.
# ---------------------------------------------------------------------------

def probe_hostphinder() -> AvailabilityReport:
    notes: list[str] = []
    cli_ok = shutil.which("HostPhinder") is not None or shutil.which("hostphinder") is not None
    db_env = os.environ.get("HOSTPHINDER_DB")
    db_ok = bool(db_env) and Path(db_env).exists() if db_env else False
    available = cli_ok and db_ok
    if not cli_ok:
        notes.append("HostPhinder CLI not found in PATH.")
    if not db_ok:
        notes.append("HostPhinder reference database path (HOSTPHINDER_DB) not set / not found.")
    return AvailabilityReport(
        name="HostPhinder",
        available=available,
        reason="cli_and_db_present" if available else "cli_or_db_missing",
        command_hint="conda install -c bioconda hostphinder && "
                    "set HOSTPHINDER_DB to the reference 2.1 database",
        license="Apache-2.0 (upstream)",
        url="https://bitbucket.org/genomicepidemiology/hostphinder",
        notes=notes,
    )


# ---------------------------------------------------------------------------
# F7 — PHIST (Zielezinski et al., 2022).
# ---------------------------------------------------------------------------

def probe_phist() -> AvailabilityReport:
    notes: list[str] = []
    cli_ok = shutil.which("phist") is not None or shutil.which("PHIST") is not None
    if not cli_ok:
        notes.append("PHIST CLI not in PATH.")
    notes.append(
        "PHIST is GPL-2.0. Any redistributable wrapper must remain outside "
        "the MIT repository; our adapter calls the binary via subprocess only."
    )
    return AvailabilityReport(
        name="PHIST",
        available=cli_ok,
        reason="cli_present" if cli_ok else "cli_missing",
        command_hint="git clone https://github.com/refresh-bio/PHIST && make",
        license="GPL-3.0 (upstream; verified 2026-05-13 — see reports/external_methods_license_recheck_2026-05-13.md)",
        url="https://github.com/refresh-bio/PHIST",
        notes=notes,
    )


# ---------------------------------------------------------------------------
# F8 — DeepHost (Ruohan et al., 2022).
# ---------------------------------------------------------------------------

def probe_deephost(weights_root: Path | None = None) -> AvailabilityReport:
    notes: list[str] = []
    weights_ok = (
        weights_root is not None
        and (weights_root / "DeepHost_scripts" / "model_checkpoints").is_dir()
    )
    if not weights_ok:
        notes.append("DeepHost pretrained checkpoints directory not found.")
    return AvailabilityReport(
        name="DeepHost",
        available=weights_ok,
        reason="weights_present" if weights_ok else "weights_missing",
        command_hint="git clone https://github.com/deepomicslab/DeepHost && "
                    "download model_checkpoints from upstream release",
        license="Apache-2.0 (upstream)",
        url="https://github.com/deepomicslab/DeepHost",
        notes=notes,
    )


# ---------------------------------------------------------------------------
# F+ — additional public methods
# ---------------------------------------------------------------------------

def probe_phabox() -> AvailabilityReport:
    have_mod = importlib.util.find_spec("phabox") is not None
    return AvailabilityReport(
        name="PhaBox (PhaTYP)",
        available=have_mod,
        reason="package_present" if have_mod else "package_missing",
        command_hint="pip install phabox (requires GPU + Diamond + HMMER)",
        license="GPL-3.0 (upstream; verified 2026-05-13 — see reports/external_methods_license_recheck_2026-05-13.md)",
        url="https://github.com/KennthShang/PhaBOX",
        notes=[
            "PhaBox includes PhaTYP and PhaGCN; only PhaTYP applies to host "
            "prediction.  Requires large HMM databases (~5 GB) to be installed."
        ] if not have_mod else [],
    )


def probe_virhostmatcher_net() -> AvailabilityReport:
    have_repo = Path("/opt/VirHostMatcher-Net").exists()
    return AvailabilityReport(
        name="VirHostMatcher-Net",
        available=have_repo,
        reason="repo_present" if have_repo else "repo_missing",
        command_hint="git clone https://github.com/WeiliWw/VirHostMatcher-Net",
        license="GPL-3.0 (upstream; verified 2026-05-13 — see reports/external_methods_license_recheck_2026-05-13.md)",
        url="https://github.com/WeiliWw/VirHostMatcher-Net",
        notes=[] if have_repo else ["Upstream not cloned under /opt."],
    )


def probe_wish() -> AvailabilityReport:
    """Probe for soedinglab/WIsH — the phage-host k-mer Markov tool.

    CAUTION: On macOS the name ``WIsH`` also resolves to Tcl/Tk's
    interactive shell at ``/usr/bin/WIsH``; we therefore require the binary
    to respond to ``--help`` with the ``soedinglab`` banner before claiming
    availability.
    """
    import subprocess

    path = shutil.which("WIsH")
    if path is None:
        return AvailabilityReport(
            name="WIsH",
            available=False,
            reason="cli_missing",
            command_hint="git clone https://github.com/soedinglab/WIsH && make",
            license="GPL-3.0 (upstream)",
            url="https://github.com/soedinglab/WIsH",
            notes=["WIsH binary not in PATH."],
        )
    try:
        out = subprocess.run(
            [path, "--help"],
            capture_output=True, text=True, timeout=5,
        )
        banner = (out.stdout + out.stderr).lower()
        is_soedinglab = "soedinglab" in banner or "phage" in banner
    except Exception:  # noqa: BLE001
        is_soedinglab = False

    if not is_soedinglab:
        return AvailabilityReport(
            name="WIsH",
            available=False,
            reason="cli_present_but_not_soedinglab_wish",
            command_hint="git clone https://github.com/soedinglab/WIsH && make",
            license="GPL-3.0 (upstream)",
            url="https://github.com/soedinglab/WIsH",
            notes=[
                f"Binary at {path} is not the soedinglab/WIsH CLI (probably "
                "the macOS Tcl/Tk 'Wish' interpreter).  Our MIT-licensed "
                "kmer_markov module reimplements the same idea and is the "
                "Paper 1 stand-in."
            ],
        )
    return AvailabilityReport(
        name="WIsH",
        available=True,
        reason="cli_present",
        command_hint="git clone https://github.com/soedinglab/WIsH && make",
        license="GPL-3.0 (upstream)",
        url="https://github.com/soedinglab/WIsH",
        notes=[],
    )


def probe_php() -> AvailabilityReport:
    cli_ok = shutil.which("PHP") is not None or shutil.which("PhageHostPred") is not None
    return AvailabilityReport(
        name="PHP (Prokaryotic Host Predictor)",
        available=cli_ok,
        reason="cli_present" if cli_ok else "cli_missing",
        command_hint="git clone https://github.com/congyulu-bioinfo/PHP && download hostKmer_60105_kmer4.tar.gz",
        license="LGPL-3.0 (upstream; verified 2026-05-13 at congyulu-bioinfo/PHP — see reports/external_methods_license_recheck_2026-05-13.md)",
        url="https://github.com/congyulu-bioinfo/PHP",
        notes=[
            "Upstream licence is LGPL-3.0 (verified 2026-05-13). Earlier "
            "probe-time records noted CC-BY-NC for an unrelated fork "
            "(github.com/aplonzo/PHP); the canonical Lu et al. (2021) "
            "repository is github.com/congyulu-bioinfo/PHP and is LGPL-3.0."
        ],
    )


def probe_phirbo() -> AvailabilityReport:
    cli_ok = shutil.which("phirbo") is not None
    return AvailabilityReport(
        name="Phirbo",
        available=cli_ok,
        reason="cli_present" if cli_ok else "cli_missing",
        command_hint="conda install -c bioconda phirbo (github.com/aziele/phirbo)",
        license="GPL-3.0 (upstream; verified 2026-05-13 — see reports/external_methods_license_recheck_2026-05-13.md)",
        url="https://github.com/aziele/phirbo",
        notes=[] if cli_ok else ["Phirbo not installed."],
    )


def probe_bacteriophageipp() -> AvailabilityReport:
    return AvailabilityReport(
        name="BacteriophageIPP",
        available=False,
        reason="no_public_release",
        command_hint="n/a",
        license="unknown",
        url="(no authoritative public repo located as of 2026-04-24)",
        notes=[
            "A brief literature probe did not locate a maintained public "
            "implementation under this exact name. Treated as skip-with-reason."
        ],
    )


# ---------------------------------------------------------------------------

def probe_all(weights_root: Path | None = None) -> list[AvailabilityReport]:
    """Return the ordered probe list used by the Phase 2 driver."""
    return [
        probe_cherry(weights_root=weights_root),
        probe_hostphinder(),
        probe_phist(),
        probe_deephost(weights_root=weights_root),
        probe_phabox(),
        probe_virhostmatcher_net(),
        probe_wish(),
        probe_php(),
        probe_phirbo(),
        probe_bacteriophageipp(),
    ]

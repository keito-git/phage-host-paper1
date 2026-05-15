# MIT License. See LICENSE in repository root.
"""F5–F8 + F+ — Availability survey for external host-range methods.

This script does **not** run any external tool.  It only probes the host for
availability (installed CLIs, pretrained weights) and writes a structured
report so the findings report can faithfully record what was and was not
attempted.  The Phase 2 decision is: "we tried to run everything; here is
what was possible on the current hardware."
"""
from __future__ import annotations

from common import ensure_path

ensure_path()

import argparse
import dataclasses
import json

from src.baselines.adapters.external_methods import probe_all
from src.config import REPORTS_DIR, ensure_dirs


def main() -> dict:
    ensure_dirs()
    reports = probe_all()
    payload = [dataclasses.asdict(r) for r in reports]
    out_path = REPORTS_DIR / "f_plus_survey.json"
    out_path.write_text(json.dumps(payload, indent=2))

    md_lines = [
        "# F5–F8 + F+ — External method availability probe",
        "",
        "Each method was probed without triggering downloads. `available=True`",
        "means both the executable (or Python package) and pretrained weights",
        "(where applicable) were found locally.",
        "",
        "| method | available | reason | licence | upstream |",
        "|---|---|---|---|---|",
    ]
    for r in reports:
        md_lines.append(
            f"| {r.name} | {'YES' if r.available else 'NO'} | {r.reason} | "
            f"{r.license} | {r.url} |"
        )
    md_lines.append("")
    md_lines.append("## Detailed notes")
    md_lines.append("")
    for r in reports:
        md_lines.append(f"### {r.name}")
        md_lines.append(f"- Available: **{r.available}**")
        md_lines.append(f"- Reason: {r.reason}")
        md_lines.append(f"- Upstream: {r.url}")
        md_lines.append(f"- Licence: {r.license}")
        md_lines.append(f"- Install hint: `{r.command_hint}`")
        if r.notes:
            md_lines.append("- Notes:")
            for n in r.notes:
                md_lines.append(f"  - {n}")
        md_lines.append("")
    (REPORTS_DIR / "f_plus_survey.md").write_text("\n".join(md_lines))

    return {
        "path_json": str(out_path),
        "path_md": str(REPORTS_DIR / "f_plus_survey.md"),
        "n_available": sum(1 for r in reports if r.available),
        "n_probed": len(reports),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(main(), indent=2))

"""Prepare self-contained HESS submission sources from the auditable author files."""

from __future__ import annotations

import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = ROOT / "paper"
GENERATED_DIR = PAPER_DIR / "generated"
SUBMISSION_DIR = ROOT / "submission" / "hess"

MAIN_FIGURES = [
    ("figures/fig_protocol_timeline.pdf", "f01.pdf"),
    ("figures/architecture_tikz.pdf", "f02.pdf"),
    ("figures/fig1_scarcity_effects.pdf", "f03.pdf"),
    ("figures/figS5_subset_replication.pdf", "f04.pdf"),
    ("figures/fig2_conditional_coverage.pdf", "f05.pdf"),
    ("figures/figS3_high_flow_thresholds.pdf", "f06.pdf"),
    ("figures/fig3_method_comparison.pdf", "f07.pdf"),
]

SUPPLEMENT_FIGURES = [
    ("figures/figS1_basin_endpoint_effects.pdf", "fs01.pdf"),
    ("figures/figS2_climate_quartiles.pdf", "fs02.pdf"),
    ("figures/figS4_basin_calibration.pdf", "fs03.pdf"),
    ("figures/figS6_equal_updates.pdf", "fs04.pdf"),
    ("figures/figS7_architecture_sensitivity.pdf", "fs05.pdf"),
]

SHORT_SUMMARY = (
    "Reliable streamflow forecasts need both accurate predictions and trustworthy uncertainty "
    "ranges. Using matched experiments across 250 United States basins, we show that shortening "
    "supervised records from 15 years to 1 year strongly reduces prediction skill and also weakens "
    "interval coverage, especially at high flows. Conformal calibration restores overall coverage "
    "but not all high-flow reliability, so data volume, calibration, and conditional diagnostics "
    "must be assessed together."
)


def read_required(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Required submission input not found: {path}")
    return path.read_text(encoding="utf-8")


def parse_result_macros(text: str) -> dict[str, str]:
    macros: dict[str, str] = {}
    pattern = re.compile(r"^\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}$")
    for line in text.splitlines():
        if not line.strip():
            continue
        match = pattern.fullmatch(line)
        if match is None:
            raise ValueError(f"Unsupported generated macro definition: {line}")
        name, value = match.groups()
        if name in macros:
            raise ValueError(f"Duplicate generated macro: {name}")
        macros[name] = value
    if not macros:
        raise ValueError("No generated result macros were found")
    return macros


def expand_result_macros(text: str, macros: dict[str, str]) -> str:
    for name in sorted(macros, key=len, reverse=True):
        value = macros[name]
        text = text.replace(f"\\{name}\\ ", value + " ")
        text = re.sub(
            rf"\\{re.escape(name)}(?![A-Za-z@])",
            lambda _: value,
            text,
        )
    unresolved = sorted(set(re.findall(r"\\([A-Z][A-Za-z]+)", text)) & set(macros))
    if unresolved:
        raise ValueError(f"Unexpanded generated macros: {unresolved}")
    return text


def inline_generated_rows(text: str) -> str:
    pattern = re.compile(
        r"(?m)^(?P<indent>[ \t]*)\\inputtablerows\{generated/(?P<name>[^}]+)\}[ \t]*$"
    )

    def replacement(match: re.Match[str]) -> str:
        fragment = read_required(GENERATED_DIR / match.group("name")).rstrip("\n")
        indent = match.group("indent")
        return "\n".join(indent + line for line in fragment.splitlines())

    return pattern.sub(replacement, text)


def remove_authoring_helpers(text: str, *, hess_main: bool) -> str:
    text = re.sub(
        r"(?m)^\\input\{generated/result_macros\.tex\}[ \t]*\r?\n",
        "",
        text,
    )
    text = re.sub(
        r"(?ms)^\\makeatletter\s*\n"
        r"\\newcommand\{\\inputtablerows\}\[1\]\{\\@@input\{#1\}\}\s*\n"
        r"\\makeatother\s*\n",
        "",
        text,
    )
    if hess_main:
        text = re.sub(r"(?m)^\\usepackage(?:\[[^]]*\])?\{[^}]+\}[ \t]*\r?\n", "", text)
        text = text.replace(r"\begin{enumerate}[nosep]", r"\begin{enumerate}")
        text = text.replace(r"\begin{itemize}[nosep]", r"\begin{itemize}")
    return text


def flatten_source(source_name: str, *, hess_main: bool) -> str:
    text = read_required(PAPER_DIR / source_name)
    macros = parse_result_macros(read_required(GENERATED_DIR / "result_macros.tex"))
    text = remove_authoring_helpers(text, hess_main=hess_main)
    text = expand_result_macros(text, macros)
    text = inline_generated_rows(text)
    if hess_main:
        references = read_required(PAPER_DIR / "references.tex").rstrip("\n")
        text = text.replace(r"\input{references.tex}", references)
    forbidden = [r"\inputtablerows", r"\input{generated/", r"\newcommand"]
    if hess_main:
        forbidden.extend([r"\input{", r"\usepackage"])
    remaining = [token for token in forbidden if token in text]
    if remaining:
        raise ValueError(f"Submission source still contains authoring helpers: {remaining}")
    return text.rstrip() + "\n"


def copy_numbered_figures(text: str, mappings: list[tuple[str, str]]) -> str:
    for source_name, submission_name in mappings:
        source = PAPER_DIR / source_name
        if not source.exists() or source.stat().st_size == 0:
            raise FileNotFoundError(f"Required figure not found: {source}")
        shutil.copy2(source, SUBMISSION_DIR / submission_name)
        text = text.replace(source_name, submission_name)
    return text


def abstract_text(manuscript: str) -> str:
    match = re.search(r"\\begin\{abstract\}\s*(.*?)\s*\\end\{abstract\}", manuscript, re.DOTALL)
    if match is None:
        raise ValueError("Abstract not found in manuscript source")
    text = re.sub(r"\\%", "%", match.group(1))
    text = text.replace("~", " ")
    text = re.sub(r"\\(?:url|citep|citet)\{[^}]+\}", "", text)
    text = re.sub(r"\\[A-Za-z]+", "", text)
    text = re.sub(r"[{}$]", "", text)
    return re.sub(r"\s+", " ", text).strip() + "\n"


def main() -> None:
    if len(SHORT_SUMMARY) > 500:
        raise ValueError("HESS short summary exceeds the 500-character limit")
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    for pattern in ("f[0-9][0-9].pdf", "fs[0-9][0-9].pdf"):
        for path in SUBMISSION_DIR.glob(pattern):
            path.unlink()
    manuscript = flatten_source("manuscript.tex", hess_main=True)
    manuscript = copy_numbered_figures(manuscript, MAIN_FIGURES)
    supplement = flatten_source("supplement.tex", hess_main=False)
    supplement = copy_numbered_figures(supplement, SUPPLEMENT_FIGURES)

    (SUBMISSION_DIR / "manuscript.tex").write_text(manuscript, encoding="utf-8")
    (SUBMISSION_DIR / "supplement.tex").write_text(supplement, encoding="utf-8")
    (SUBMISSION_DIR / "abstract.txt").write_text(abstract_text(manuscript), encoding="utf-8")
    (SUBMISSION_DIR / "short_summary.txt").write_text(SHORT_SUMMARY + "\n", encoding="utf-8")

    support_files = [
        "copernicus.cls",
        "copernicus.cfg",
        "copernicus.bst",
        "pdfscreen.sty",
        "pdfscreencop.sty",
    ]
    for name in support_files:
        source = PAPER_DIR / name
        if source.exists():
            shutil.copy2(source, SUBMISSION_DIR / name)

    print(f"Submission sources and numbered figures written to {SUBMISSION_DIR}")


if __name__ == "__main__":
    main()

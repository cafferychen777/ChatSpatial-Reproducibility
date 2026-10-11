#!/usr/bin/env python3
"""Build the submission support files: Supplementary Tables 5 and 6, the Supplementary Tables workbook,
and the Source Data workbook and its CSV sheets.

Outputs:
  supplementary_tables/Supplementary_Table_5_Analysis_Paradigm_Comparison.csv
  supplementary_tables/Supplementary_Table_6_Representative_Failure_Cases.csv
  supplementary_tables/Supplementary_Tables.xlsx
  source_data.xlsx
  source_data.zip
  source_data/*.csv
"""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

import pandas as pd

PAPER_DIR = Path(__file__).resolve().parents[1]  # repository root
ROUND1 = PAPER_DIR / "experiments" / "round1" / "results"
H2H = PAPER_DIR / "experiments" / "refusals_and_head_to_head" / "results" / "h2h_v159"
SUPP_TABLE_DIR = PAPER_DIR / "supplementary_tables"
SOURCE_DIR = PAPER_DIR / "source_data"

EXCEL_ROW_LIMIT = 1_048_576


TABLE_5_ROWS = [
    {
        "Workflow Aspect": "Primary Unit of Interaction",
        "ChatSpatial (Conversational Orchestration)": 'A high-level scientific goal (e.g., "Find spatial domains")',
        "GUI Platforms (e.g., Loupe Browser, Giotto Viewer)": "A menu-driven selection or button click (e.g., select clustering algorithm)",
        "Direct Programming (Python/R)": "A specific line of code (e.g., sc.tl.leiden(...))",
    },
    {
        "Workflow Aspect": "Cognitive Focus of User",
        "ChatSpatial (Conversational Orchestration)": "Biological hypothesis and interpretation of results",
        "GUI Platforms (e.g., Loupe Browser, Giotto Viewer)": "Navigation of software interface and understanding of available options",
        "Direct Programming (Python/R)": "Correct syntax, library versions, and data object manipulation",
    },
    {
        "Workflow Aspect": "Handling Multi-Step Analysis",
        "ChatSpatial (Conversational Orchestration)": "Fluid conversation where context is maintained by the LLM",
        "GUI Platforms (e.g., Loupe Browser, Giotto Viewer)": "Limited to workflows supported by the platform; multi-tool chaining often unavailable",
        "Direct Programming (Python/R)": "Manual scripting, requiring the user to manage state and data flow",
    },
    {
        "Workflow Aspect": "Cross-Language Integration (e.g., Python to R)",
        "ChatSpatial (Conversational Orchestration)": "Encapsulated within wrappers and recorded in the workflow log. The framework manages data conversion without user-written export scripts.",
        "GUI Platforms (e.g., Loupe Browser, Giotto Viewer)": "Generally confined to a single ecosystem (Python or R); interoperability not supported",
        "Direct Programming (Python/R)": "Manual and error-prone. Requires writing explicit export/import scripts (e.g., via zellkonverter).",
    },
    {
        "Workflow Aspect": "Parameter Selection",
        "ChatSpatial (Conversational Orchestration)": "Schema descriptions expose curated defaults and method guidance for researcher review",
        "GUI Platforms (e.g., Loupe Browser, Giotto Viewer)": "Exposed through interface controls with platform-set defaults",
        "Direct Programming (Python/R)": "User must manually look up and specify parameters in code",
    },
    {
        "Workflow Aspect": "Audit Trail",
        "ChatSpatial (Conversational Orchestration)": "Versioned schemas, wrappers, tool calls, parameters, and environment",
        "GUI Platforms (e.g., Loupe Browser, Giotto Viewer)": "Session logs or exported configuration files, where available",
        "Direct Programming (Python/R)": "Scripts, package versions, lockfiles, and environment configuration",
    },
    {
        "Workflow Aspect": "Role of Foundational Libraries (Scanpy, Seurat, etc.)",
        "ChatSpatial (Conversational Orchestration)": "Act as versioned execution engines invoked through auditable schemas.",
        "GUI Platforms (e.g., Loupe Browser, Giotto Viewer)": "Bundled internally; user interacts through the GUI layer",
        "Direct Programming (Python/R)": "Act as the direct, code-level interface for the user.",
    },
]

TABLE_6_ROWS = [
    {
        "No.": 1,
        "Category": "Missing data prerequisite",
        "Trigger": "Spatial visualization on object without spatial coordinates",
        "Error Type": "DataNotFoundError",
        "System Response (abridged)": "Spatial coordinates not found in adata.obsm['spatial']",
        "Source": "Code path",
    },
    {
        "No.": 2,
        "Category": "Hallucinated method name",
        "Trigger": 'No-schema LLM produces "method": "deep_learning" for spatial domains',
        "Error Type": "Pydantic ValidationError",
        "System Response (abridged)": "Input should be 'spagcn', 'leiden', 'louvain', 'stagate', 'graphst' or 'banksy'",
        "Source": "Ablation: 29/30 trials",
    },
    {
        "No.": 3,
        "Category": "Case-sensitivity mismatch",
        "Trigger": 'No-schema LLM sends "RCTD" instead of "rctd"',
        "Error Type": "Pydantic ValidationError",
        "System Response (abridged)": "Input should be 'flashdeconv', 'cell2location', 'rctd', ...",
        "Source": "Ablation: 30/30 trials",
    },
    {
        "No.": 4,
        "Category": "Valid but unavailable method",
        "Trigger": "Bare-schema LLM selects cell2location (not installed)",
        "Error Type": "DependencyError",
        "System Response (abridged)": "Method 'cell2location' requires: cell2location. Available: flashdeconv, destvi, ..., card (flashdeconv recommended)",
        "Source": "Ablation e2e: 6/10 trials",
    },
    {
        "No.": 5,
        "Category": "Ambiguous intent",
        "Trigger": '"Analyze the data" without specifying analytical goal',
        "Error Type": "Clarification request",
        "System Response (abridged)": "System asks user to specify analysis type (e.g., spatial domains, differential expression)",
        "Source": "Test scenario 18",
    },
    {
        "No.": 6,
        "Category": "Out-of-scope request",
        "Trigger": '"Write Python code to download a file"',
        "Error Type": "Scope redirection",
        "System Response (abridged)": "Politely redirected to spatial analysis scope",
        "Source": "Test scenario 28",
    },
]

TABLE_1_ROWS = [
    ("Preprocessing", "supported", "supported", "supported", "supported", "supported", "supported"),
    ("Multi-Sample Integration", "supported", "supported", "supported", "supported", "supported", "supported"),
    ("Cell Type Annotation", "supported", "supported", "supported", "supported", "supported", "supported"),
    ("Differential Expression", "supported", "supported", "supported", "supported", "supported", "supported"),
    ("Spatial Domain ID", "commonly used or wrapper-recommended", "supported", "supported", "supported", "supported", "supported"),
    ("Spatial Variable Genes", "supported", "supported", "supported", "supported", "supported", "supported"),
    ("Cell Communication", "supported", "supported", "commonly used or wrapper-recommended", "commonly used or wrapper-recommended", "supported", "supported"),
    ("Deconvolution", "commonly used or wrapper-recommended", "optional or assay-dependent", "not required/not applicable", "not required/not applicable", "optional or assay-dependent", "optional or assay-dependent"),
    ("CNV Analysis", "supported", "supported", "supported", "supported", "supported", "supported"),
    ("Trajectory Analysis", "supported", "supported", "commonly used or wrapper-recommended", "commonly used or wrapper-recommended", "supported", "supported"),
    ("RNA Velocity", "supported", "supported", "supported", "supported", "supported", "supported"),
]


def require_path(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def sheet_name(name: str, used: set[str]) -> str:
    cleaned = re.sub(r"[\[\]\:\*\?\/\\]", "_", name)[:31]
    base = cleaned
    idx = 2
    while cleaned in used:
        suffix = f"_{idx}"
        cleaned = f"{base[:31 - len(suffix)]}{suffix}"
        idx += 1
    used.add(cleaned)
    return cleaned


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_supplementary_tables() -> list[Path]:
    SUPP_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    table5 = pd.DataFrame(TABLE_5_ROWS)
    table6 = pd.DataFrame(TABLE_6_ROWS)

    table5_path = SUPP_TABLE_DIR / "Supplementary_Table_5_Analysis_Paradigm_Comparison.csv"
    table6_path = SUPP_TABLE_DIR / "Supplementary_Table_6_Representative_Failure_Cases.csv"
    write_csv(table5_path, table5)
    write_csv(table6_path, table6)

    table_paths = [
        SUPP_TABLE_DIR / "Supplementary_Table_1_Effect_Sizes.csv",
        SUPP_TABLE_DIR / "Supplementary_Table_2_Integrated_Methods.csv",
        SUPP_TABLE_DIR / "Supplementary_Table_3_Test_Scenarios.csv",
        SUPP_TABLE_DIR / "Supplementary_Table_4_AI_Agent_Comparison.csv",
        table5_path,
        table6_path,
        SUPP_TABLE_DIR / "Supplementary_Table_7_Deconvolution_Method_Concordance.csv",
    ]
    # Supplementary Table 8 (experiment B) is backed by several CSV files.
    expb_paths = sorted(SUPP_TABLE_DIR.glob("Supplementary_Table_expB_*.csv"))

    workbook = SUPP_TABLE_DIR / "Supplementary_Tables.xlsx"
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        for idx, path in enumerate(table_paths, start=1):
            df = pd.read_csv(require_path(path))
            df.to_excel(writer, sheet_name=f"Table_{idx}", index=False)
        for path in expb_paths:
            short = path.stem.replace("Supplementary_Table_expB_", "")
            pd.read_csv(path).to_excel(writer, sheet_name=f"Table_8_{short}"[:31], index=False)

    return table_paths + expb_paths + [workbook]


def table1_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        TABLE_1_ROWS,
        columns=[
            "Module",
            "Visium",
            "Visium HD",
            "MERFISH/Xenium",
            "seqFISH/CosMx",
            "Slide-seq",
            "STARmap",
        ],
    )


def read_jsonl_summary(path: Path) -> pd.DataFrame:
    rows = []
    with require_path(path).open() as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            rows.append(
                {
                    "line": line_no,
                    "model": record.get("model"),
                    "prompt_id": record.get("prompt_id"),
                    "replicate": record.get("replicate"),
                    "tool_name": record.get("tool_name"),
                    "parse_success": record.get("parse_success"),
                    "pydantic_valid": record.get("pydantic_valid"),
                    "error": record.get("error"),
                }
            )
    return pd.DataFrame(rows)


def source_data_sheets() -> list[tuple[str, pd.DataFrame, str, str]]:
    sheets: list[tuple[str, pd.DataFrame, str, str]] = []

    notes = pd.DataFrame(
        [
            {
                "item": "Figure 1",
                "note": "Conceptual architecture schematic; no quantitative plotted source data.",
            },
            {
                "item": "Figure 5 H&E backgrounds",
                "note": "Histology image files are figure image sources; plotted spot coordinates and quantitative overlays are provided in Fig5_spatial_maps.",
            },
            {
                "item": "Figure 6 spatial maps",
                "note": "Per-spot coordinates, tumor fractions, CNV scores, and CNV subclone labels are provided in Fig6_spatial_maps.",
            },
        ]
    )
    sheets.append(("README", notes, "generated", "Notes for source-data workbook."))

    sheets.append(("Table1_platform", table1_dataframe(), "main.tex", "Main text platform compatibility table."))

    # Sheets are labelled by the figure numbers of the manuscript: Figure 2 DLPFC
    # head-to-head (a-b), ablation and cross-system (c-k); Figure 3 backend study; Figure 4 parameter
    # robustness; Figure 5 OSCC case study; Figure 6 HGSOC case study.
    sheets.append(("Fig2ab_h2h_trials", pd.read_csv(require_path(H2H / "fig2ab_trials.csv")), "fig2ab_trials.csv", "Figure 2a-b DLPFC head-to-head: per-trial ARI and NMI for the four arms (12 sections x 10 replicates)."))
    sheets.append(("Fig2ab_h2h_sections", pd.read_csv(require_path(H2H / "fig2ab_sections.csv")), "fig2ab_sections.csv", "Figure 2a-b DLPFC head-to-head: per-section mean ARI (points in a) and cross-replicate ARI."))
    sheets.append(("Fig2b_h2h_summary", pd.read_csv(require_path(H2H / "fig2b_summary.csv")), "fig2b_summary.csv", "Figure 2b: system means with 95% section-bootstrap intervals."))
    sheets.append(("Fig2c_invocation", pd.read_csv(require_path(ROUND1 / "ablation" / "invocation" / "ablation_invocation_results.csv")), "ablation_invocation_results.csv", "Figure 2c invocation ablation."))
    sheets.append(("Fig2de_ablation_e2e", pd.read_csv(require_path(ROUND1 / "ablation" / "e2e" / "ablation_e2e_results.csv")), "ablation_e2e_results.csv", "Figure 2d-e execution rates."))
    sheets.append(("Fig2e_concordance", pd.read_csv(require_path(ROUND1 / "ablation" / "analysis" / "ablation_metrics_bootstrap.csv")), "ablation_metrics_bootstrap.csv", "Figure 2e concordance."))
    sheets.append(("Fig2fg_sensitivity", pd.read_csv(require_path(ROUND1 / "sensitivity" / "prompt_sensitivity_results.csv")), "prompt_sensitivity_results.csv", "Figure 2f-g prompt sensitivity summary."))
    sheets.append(("Fig2fg_sens_raw", read_jsonl_summary(ROUND1 / "sensitivity" / "prompt_sensitivity_raw.jsonl"), "prompt_sensitivity_raw.jsonl", "Figure 2f-g raw prompt sensitivity trials."))
    sheets.append(("Fig2hi_cross_system", pd.read_csv(require_path(ROUND1 / "cross_system" / "analysis" / "cross_system_metrics.csv")), "cross_system_metrics.csv", "Figure 2h-i cross-system metrics."))
    sheets.append(("Fig2j_e2e_summary", pd.read_csv(require_path(ROUND1 / "e2e_benchmark" / "e2e_benchmark_results.csv")), "e2e_benchmark_results.csv", "Figure 2j end-to-end benchmark summary."))

    # Figures 3 and 4 source data are written by figures/figureB_backends.py and
    # figures/figureA_robustness.py into source_data/ and are read here unchanged.
    # Figures 5 and 6 come from the recorded ChatSpatial runs of the case studies
    # (experiments/case_studies/oscc, experiments/case_studies/hgsoc). Their source-data CSVs
    # were written from those outputs (Figure 5: oscc/scripts/oscc157_postrun.py) and are read
    # here unchanged.
    for name, description in [
        ("Fig3a_backends", "Figure 3a output-verified completion and tool-selection accuracy per backend."),
        ("Fig3b_outcomes", "Figure 3b trial outcomes by schema information and constraint mechanism."),
        ("Fig3c_native_vs_json", "Figure 3c odds ratios for native tool calling versus JSON output."),
        ("Fig3d_description_ablation", "Figure 3d description ablation."),
        ("Fig3e_invocation_determinism", "Figure 3e invocation determinism."),
        ("Fig3f_method_choice", "Figure 3f method choice for open-ended requests."),
        ("Fig4a_sensitivity", "Figure 4a ARI range across schema-valid configurations."),
        ("Fig4b_strategies", "Figure 4b mean ARI under four strategies."),
        ("Fig4c_noise", "Figure 4c ARI retained under count thinning and spot removal."),
        ("Fig4d_platforms", "Figure 4d mean ARI of each method on each platform."),
        ("Fig5_spatial_maps", "Figure 5a-c,g spot-level overlays."),
        ("Fig5d_enrichment", "Figure 5d heatmap."),
        ("Fig5e_consensus", "Figure 5e bars."),
        ("Fig5e_consensus_CI", "Figure 5e confidence intervals."),
        ("Fig5f_cellchat", "Figure 5f dot plot."),
        ("Fig5h_morans", "Figure 5h Moran's I values."),
        ("Fig5i_marker_genes", "Figure 5i marker gene source."),
        ("Fig5i_DE_all_genes", "Differential expression table underlying marker summaries."),
        ("Fig6_spatial_maps", "Figure 6a-b per-spot maps."),
        ("Fig6cd_summary", "Figure 6c-d summary values."),
        ("Fig6ef_svg_results", "Figure 6e-f Moran's I values."),
        ("Fig6f_effect_size", "Figure 6f effect size."),
    ]:
        path = require_path(SOURCE_DIR / f"{name}.csv")
        sheets.append((name, pd.read_csv(path), str(path.relative_to(PAPER_DIR)), description))

    for idx in range(1, 8):
        filename = f"Supplementary_Table_{idx}_" + {
            1: "Effect_Sizes.csv",
            2: "Integrated_Methods.csv",
            3: "Test_Scenarios.csv",
            4: "AI_Agent_Comparison.csv",
            5: "Analysis_Paradigm_Comparison.csv",
            6: "Representative_Failure_Cases.csv",
            7: "Deconvolution_Method_Concordance.csv",
        }[idx]
        path = require_path(SUPP_TABLE_DIR / filename)
        sheets.append((f"Supp_Table_{idx}", pd.read_csv(path), str(path.relative_to(PAPER_DIR)), f"Supplementary Table {idx}."))

    return sheets


def write_source_data() -> list[Path]:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    sheets = source_data_sheets()

    metadata_rows = []
    used_names: set[str] = set()
    source_csv_paths: list[Path] = []

    workbook = PAPER_DIR / "source_data.xlsx"
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        for requested_name, df, source, description in sheets:
            if len(df) >= EXCEL_ROW_LIMIT:
                raise ValueError(f"{requested_name} has {len(df)} rows, exceeding Excel row limit")

            name = sheet_name(requested_name, used_names)
            metadata_rows.append(
                {
                    "sheet": name,
                    "rows": len(df),
                    "columns": len(df.columns),
                    "source": source,
                    "description": description,
                }
            )
            out_csv = SOURCE_DIR / f"{name}.csv"
            write_csv(out_csv, df)
            source_csv_paths.append(out_csv)
            df.to_excel(writer, sheet_name=name, index=False)

        metadata = pd.DataFrame(metadata_rows)
        metadata_name = sheet_name("Index", used_names)
        metadata.to_excel(writer, sheet_name=metadata_name, index=False)
        index_csv = SOURCE_DIR / f"{metadata_name}.csv"
        write_csv(index_csv, metadata)
        source_csv_paths.append(index_csv)

    zip_path = PAPER_DIR / "source_data.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in source_csv_paths:
            zf.write(path, arcname=path.relative_to(PAPER_DIR))

    return [workbook, zip_path, *source_csv_paths]


def main() -> None:
    supplementary_outputs = write_supplementary_tables()
    source_outputs = write_source_data()
    print("Supplementary outputs:")
    for path in supplementary_outputs:
        print(f"  {path.relative_to(PAPER_DIR)}")
    print("Source data outputs:")
    for path in source_outputs[:2]:
        print(f"  {path.relative_to(PAPER_DIR)}")
    print(f"  source_data/*.csv ({len(source_outputs) - 2} files)")


if __name__ == "__main__":
    main()

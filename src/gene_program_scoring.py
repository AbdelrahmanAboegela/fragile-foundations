"""Load Hallmark gene sets and score predicted expression vectors per program.

Uses gseapy to pull the live Hallmark (H) collection from MSigDB rather than
a hardcoded gene list frozen from memory — see configs/gene_programs.yaml
for why that matters.
"""
from __future__ import annotations

from pathlib import Path

import yaml
import numpy as np
import pandas as pd
import gseapy as gp

# Anchor on the project root, not CWD — same bug class as kill_test.py
# (found 2026-09-15 when running from src/ broke every relative path here too).
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_hallmark_gene_sets(cache_path: str | None = None) -> dict[str, list[str]]:
    """Load the Hallmark collection.

    Default source: data/hallmark_enrichr.gmt, pulled live from Enrichr's
    public mirror (library=MSigDB_Hallmark_2020) via
        curl 'https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=MSigDB_Hallmark_2020'
    No login required — this is how the 5 target programs in
    configs/gene_programs.yaml were verified (49/50 sets, all 5 we need
    present). Enrichr uses human-readable names ("Hypoxia") rather than
    MSigDB's "HALLMARK_X_Y" prefix — configs/gene_programs.yaml already
    matches this file's naming.

    If a canonical MSigDB .gmt (HALLMARK_-prefixed names, downloaded
    manually from gsea-msigdb.org) is placed at
    data/h.all.v2024.1.Hs.symbols.gmt instead, that path also works — same
    gene membership, just update configs/gene_programs.yaml's names to match
    if you switch sources (don't mix naming conventions across a run).
    """
    path = Path(cache_path) if cache_path else PROJECT_ROOT / "data/hallmark_enrichr.gmt"

    if path.exists():
        gene_sets: dict[str, list[str]] = {}
        with open(path) as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                name = parts[0]
                genes = [g for g in parts[2:] if g]  # parts[1] is description/URL, often empty
                gene_sets[name] = genes
        return gene_sets

    msigdb_path = PROJECT_ROOT / "data/h.all.v2024.1.Hs.symbols.gmt"
    if msigdb_path.exists():
        return load_hallmark_gene_sets(cache_path=str(msigdb_path))

    print(f"[warn] Neither {path} nor {msigdb_path} found — falling back to gseapy's "
          "live Enrichr fetch (network dependency at run time, no local cache).")
    import gseapy as gp
    return gp.get_library(name="MSigDB_Hallmark_2020", organism="Human")


def load_program_config(path: str | None = None) -> dict:
    resolved = Path(path) if path else PROJECT_ROOT / "configs/gene_programs.yaml"
    with open(resolved) as f:
        return yaml.safe_load(f)


def score_programs(
    expression: pd.DataFrame,
    hallmark_sets: dict[str, list[str]],
    program_config: dict,
) -> pd.DataFrame:
    """expression: samples x genes (predicted or measured), gene symbols as columns.

    Returns samples x programs score matrix using the method in
    program_config['scoring_method'] (mean_expression to start).
    """
    method = program_config.get("scoring_method", "mean_expression")
    programs = program_config["programs"]

    scores = {}
    for prog_name, spec in programs.items():
        genes: set[str] = set()
        for hset in spec["hallmark_sets"]:
            genes.update(hallmark_sets.get(hset, []))
        available = [g for g in genes if g in expression.columns]
        if not available:
            print(f"[warn] program '{prog_name}': 0/{len(genes)} genes found in expression matrix — "
                  f"check gene symbol convention (HGNC vs Ensembl) and the HEST-Bench highly-variable-gene list.")
            scores[prog_name] = np.full(len(expression), np.nan)
            continue
        if method == "mean_expression":
            # log1p-normalized (audit-flagged 2026-09-16): the previous plain
            # mean-of-raw-counts is dominated by whichever gene in the set
            # happens to be most highly expressed, so "program drift" could
            # largely reflect predicted-expression-magnitude drift rather
            # than genuine multi-gene pathway-activity change. Clip negative
            # Ridge predictions to 0 first (log1p is undefined below -1, and
            # a linear head can predict small negative counts).
            vals = expression[available].clip(lower=0)
            scores[prog_name] = np.log1p(vals).mean(axis=1).values
        else:
            raise NotImplementedError(f"Scoring method '{method}' not implemented yet — see TODO in config.")

    return pd.DataFrame(scores, index=expression.index)

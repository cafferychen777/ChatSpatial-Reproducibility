#!/usr/bin/env python3
"""Experiment B analysis (rules: ../expB_rules_2026-10-08.txt, sections 2 and 4).

Input: <EXPB_DIR>/<EXPB_EXEC_DIR>/trial_outcomes.csv (from expB_execute.py); by default ../results/trial_outcomes.csv.gz.
Output: <EXPB_DIR>/<EXPB_ANALYSIS_DIR>/*.csv and expB_summary.txt; by default ../results/analysis/.
The GLMMs run in R (lme4; brglm2 where separation occurs) through Rscript.
"""
import itertools
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import expB_common as C  # noqa: E402

import os
# Default: rerun the analysis from the tables in ../results (trial_outcomes.csv.gz -> results/analysis).
# On the cluster, EXPB_DIR, EXPB_EXEC_DIR and EXPB_ANALYSIS_DIR point at the execution outputs.
RESULTS = Path(__file__).resolve().parents[1] / "results"
EXPB = Path(os.environ.get("EXPB_DIR", RESULTS))
EXEC_DIR = os.environ.get("EXPB_EXEC_DIR", "exec" if "EXPB_DIR" in os.environ else ".")
OUT = EXPB / os.environ.get("EXPB_ANALYSIS_DIR", "analysis")
OUT.mkdir(parents=True, exist_ok=True)
B = 10000
SEED = 20261008
OPEN_WEIGHT_TAG = "ow:"  # labels of open-weight backends start with this prefix
VERSION_PAIRS = [("claude-haiku-4-5-20251001", "claude-haiku-5-5"), ("gpt-6-sol", "gpt-6.1-sol")]
CONSTRAINED = {p["id"]: p["constrained_params"] for p in C.TEST_PROMPTS}

lines = []


def say(s=""):
    print(s)
    lines.append(str(s))


def boot_rate(df, y, cluster="prompt_id", b=B, seed=SEED):
    """Cluster bootstrap over prompts: resample prompts, then trials within each prompt."""
    rng = np.random.default_rng(seed)
    groups = [g[y].astype(float).values for _, g in df.groupby(cluster)]
    if not groups:
        return np.nan, np.nan, np.nan
    est = df[y].astype(float).mean()
    bs = np.empty(b)
    k = len(groups)
    for i in range(b):
        idx = rng.integers(0, k, k)
        vals = np.concatenate([rng.choice(groups[j], len(groups[j])) for j in idx])
        bs[i] = vals.mean()
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return est, lo, hi


def fmt(est, lo, hi, pct=True):
    if np.isnan(est):
        return "NA"
    f = (lambda x: f"{100*x:.1f}%") if pct else (lambda x: f"{x:.3f}")
    return f"{f(est)} [{f(lo)}, {f(hi)}]"


def boot_cell_stat(cells: pd.DataFrame, col: str, b=B, seed=SEED):
    """Bootstrap of a cell-level statistic, resampling prompts."""
    rng = np.random.default_rng(seed)
    by = {p: g[col].values for p, g in cells.groupby("prompt_id")}
    keys = list(by)
    est = cells[col].mean()
    bs = [np.concatenate([by[keys[j]] for j in rng.integers(0, len(keys), len(keys))]).mean() for _ in range(b)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return est, lo, hi


def constrained_value(row):
    if not (row.pydantic_valid and row.tool_correct) or not isinstance(row.effective_params, str):
        return "INVALID"
    ep = json.loads(row.effective_params)
    vals = []
    for p in CONSTRAINED[row.prompt_id]:
        v = ep.get(p)
        vals.append(v.lower() if isinstance(v, str) else v)
    return json.dumps(vals)


def main():
    outcomes = EXPB / EXEC_DIR / "trial_outcomes.csv"
    if not outcomes.exists():
        outcomes = outcomes.with_suffix(".csv.gz")
    df = pd.read_csv(outcomes)
    for c in ["tool_correct", "pydantic_valid", "completed", "silent_failure", "explicit_failure", "executed", "exec_success"]:
        df[c] = df[c].fillna(False).astype(bool)
    df["open_weight"] = df.label.str.startswith(OPEN_WEIGHT_TAG)
    main_ = df[df.experiment == "main"].copy()
    main_["format_failure"] = main_.error_class == "format_failure"
    main_["mis_selection"] = main_.error_class == "mis_selection"
    main_["cval"] = main_.apply(constrained_value, axis=1)
    backends = sorted(main_.label.unique())
    say(f"Experiment B analysis — backends: {backends}")
    say(f"main trials: {len(main_)}; per backend: {main_.label.value_counts().to_dict()}")

    # ---------------- per-cell rates ----------------
    rows = []
    for (lab, mech, info), g in main_.groupby(["label", "mechanism", "info"]):
        r = dict(label=lab, mechanism=mech, info=info, n=len(g))
        for y in ["tool_correct", "format_failure", "mis_selection", "pydantic_valid", "completed", "silent_failure", "explicit_failure"]:
            est, lo, hi = boot_rate(g, y, b=2000)
            r[y], r[y + "_lo"], r[y + "_hi"] = est, lo, hi
        rows.append(r)
    rates = pd.DataFrame(rows)
    rates.to_csv(OUT / "rates_by_backend_condition.csv", index=False)
    say("\n== Rates by backend x mechanism x information (cluster bootstrap over prompts, 2000 draws)")
    say(rates[["label", "mechanism", "info", "n", "tool_correct", "format_failure", "mis_selection", "pydantic_valid",
               "completed", "silent_failure"]].round(3).to_string(index=False))

    # ---------------- headline placeholders ----------------
    say("\n== Placeholders (native + full unless stated; 10,000-draw cluster bootstrap over prompts)")
    nf = main_[(main_.mechanism == "native") & (main_["info"] == "full")]
    ow, cl = nf[nf.open_weight], nf[~nf.open_weight]
    for name, sub in [("B3 open-weight tool-selection accuracy", ow), ("B5 open-weight output-verified completion", ow),
                      ("B6a closed tool-selection accuracy", cl), ("B6b closed output-verified completion", cl)]:
        y = "tool_correct" if "accuracy" in name else "completed"
        say(f"  {name}: {fmt(*boot_rate(sub, y))} (n={len(sub)})")
    for lab in backends:
        sub = nf[nf.label == lab]
        say(f"    {lab}: accuracy {fmt(*boot_rate(sub, 'tool_correct', b=2000))}; completion {fmt(*boot_rate(sub, 'completed', b=2000))}")
    # B7: silent failure, open-weight, minimal schema
    for mech in ["native", "json", None]:
        sub = main_[main_.open_weight & (main_["info"] == "minimal")]
        if mech:
            sub = sub[sub.mechanism == mech]
        say(f"  B7 open-weight silent failure, minimal schema, mechanism={mech or 'both'}: {fmt(*boot_rate(sub, 'silent_failure'))}"
            f" (n={len(sub)}; explicit failures {sub.explicit_failure.mean():.1%})")
    for info in ["full", "bare", "minimal"]:
        for grp, sub in [("open", main_[main_.open_weight & (main_["info"] == info)]), ("closed", main_[~main_.open_weight & (main_["info"] == info)])]:
            say(f"    silent failure {grp}-weight info={info}: {fmt(*boot_rate(sub, 'silent_failure', b=2000))};"
                f" completion {fmt(*boot_rate(sub, 'completed', b=2000))}")

    # B-MIS: error decomposition
    say("\n== B-MIS error decomposition (counts of not-correct trials)")
    dec = main_.groupby(["mechanism", "info"]).agg(n=("tool_correct", "size"), wrong=("tool_correct", lambda s: int((~s).sum())),
                                                   format_failure=("format_failure", "sum"), mis_selection=("mis_selection", "sum")).reset_index()
    say(dec.to_string(index=False))
    mis = main_[main_.mis_selection]
    if len(mis):
        say("  mis-selections (label, mechanism, info, prompt -> selected):")
        for _, r in mis.iterrows():
            say(f"    {r.label} {r.mechanism} {r["info"]} {r.prompt_id} -> {r.canonical_tool}")
    dec.to_csv(OUT / "bmis_decomposition.csv", index=False)

    # ---------------- invocation determinism ----------------
    say("\n== Invocation determinism (cells = backend x prompt x condition; call = tool + sorted raw params)")
    cells = []
    for (lab, mech, info, pid), g in main_.groupby(["label", "mechanism", "info", "prompt_id"]):
        cnt = Counter(g.call)
        modal_share = cnt.most_common(1)[0][1] / len(g)
        cells.append(dict(label=lab, mechanism=mech, info=info, prompt_id=pid, n=len(g), modal_share=modal_share,
                          identical=float(len(cnt) == 1), pass_k=float(len(cnt) == 1 and g.completed.all()),
                          open_weight=g.open_weight.iloc[0]))
    cells = pd.DataFrame(cells)
    cells.to_csv(OUT / "invocation_determinism_cells.csv", index=False)
    say(cells.groupby(["mechanism", "info"])[["modal_share", "identical", "pass_k"]].mean().round(3).to_string())
    say("  B-INV per backend (native + full): modal-call share [cluster CI]; identical-call cells; pass^10")
    binv = []
    for lab in backends:
        c = cells[(cells.label == lab) & (cells.mechanism == "native") & (cells["info"] == "full")]
        e = boot_cell_stat(c, "modal_share", b=2000)
        binv.append(dict(label=lab, modal_share=e[0], lo=e[1], hi=e[2], identical=c.identical.sum(), n_cells=len(c), pass_k=c.pass_k.mean()))
        say(f"    {lab}: {fmt(*e)}; identical {int(c.identical.sum())}/{len(c)}; pass^10 {c.pass_k.mean():.1%}")
    pd.DataFrame(binv).to_csv(OUT / "binv_per_backend.csv", index=False)

    # ---------------- cross-backend agreement on constrained params ----------------
    say("\n== Agreement on constrained parameters: sum_v pA(v) pB(v), averaged over prompts")
    dist = {k: g.cval.value_counts(normalize=True).to_dict() for k, g in main_.groupby(["label", "mechanism", "info", "prompt_id"])}

    def agree(a, b, mech, info):
        vals = []
        for pid in CONSTRAINED:
            da, db = dist.get((a, mech, info, pid)), dist.get((b, mech, info, pid))
            if da is None or db is None:
                continue
            vals.append(sum(pa * db.get(v, 0.0) for v, pa in da.items()))
        return np.mean(vals) if vals else np.nan, vals

    pair_rows = []
    for a, b in itertools.combinations(backends, 2):
        for mech in ["json", "native"]:
            for info in ["full", "bare", "minimal"]:
                m, v = agree(a, b, mech, info)
                pair_rows.append(dict(a=a, b=b, mechanism=mech, info=info, agreement=m, n_prompts=len(v)))
    pairs = pd.DataFrame(pair_rows)
    pairs.to_csv(OUT / "cross_backend_agreement.csv", index=False)
    say(pairs.groupby(["mechanism", "info"]).agreement.mean().round(3).to_string())
    for a, b in VERSION_PAIRS:
        if a in backends and b in backends:
            for mech in ["native", "json"]:
                for info in ["full", "bare", "minimal"]:
                    m, v = agree(a, b, mech, info)
                    # bootstrap over prompts
                    rng = np.random.default_rng(SEED)
                    bs = [np.mean(rng.choice(v, len(v))) for _ in range(B)]
                    say(f"  B8 version pair {a} -> {b} [{mech}/{info}]: {100*m:.1f}% [{100*np.percentile(bs,2.5):.1f}, {100*np.percentile(bs,97.5):.1f}]")

    # ---------------- description ablation ----------------
    desc = df[df.experiment == "desc"].copy()
    if len(desc):
        say("\n== B-DESC description ablation (native + full); shares among trials with a parseable call to an existing tool")
        desc["partner"] = desc.expected_tool.map(C.SWAP_PARTNER)
        desc["ok"] = desc.canonical_tool.notna() & (desc.parse_status == "ok")
        desc["followed_desc"] = desc.ok & (desc.canonical_tool == desc.partner)
        desc["followed_name"] = desc.ok & (desc.canonical_tool == desc.expected_tool)
        rows = []
        for (arm, lab), g in desc.groupby(["arm", "label"]):
            gg = g[g.ok]
            rows.append(dict(arm=arm, label=lab, n=len(g), n_ok=len(gg), followed_desc=gg.followed_desc.mean(),
                             followed_target=gg.followed_name.mean()))
        dr = pd.DataFrame(rows)
        dr.to_csv(OUT / "bdesc_by_backend.csv", index=False)
        say(dr.round(3).to_string(index=False))
        for arm in ["desc_swap", "neutral_intact", "neutral_swap"]:
            g = desc[(desc.arm == arm) & desc.ok]
            say(f"  {arm}: followed edited description {fmt(*boot_rate(g, 'followed_desc'))}; selected original target {fmt(*boot_rate(g, 'followed_name'))} (n={len(g)})")
        g = desc[(desc.arm == "desc_swap") & desc.ok]
        say("  desc_swap per prompt (followed description share): " +
            ", ".join(f"{p}={v:.2f}" for p, v in g.groupby("prompt_id").followed_desc.mean().items()))

    # ---------------- method-choice bias ----------------
    bias = df[df.experiment == "bias"].copy()
    if len(bias):
        say("\n== B-BIAS method choice on open-ended prompts (native + full); executed method after defaults")
        pm = {p["id"]: p["method_param"] for p in C.BIAS_PROMPTS}
        defaults_text = {"bias_domains": "spagcn", "bias_deconv": "flashdeconv", "bias_svg": "sparkx", "bias_ccc": "fastccc",
                         "bias_autocorr": "neighborhood", "bias_traj": "cellrank", "bias_cnv": "infercnvpy"}

        def meth(r):
            if r.parse_status != "ok" or r.canonical_tool != r.expected_tool:
                return "NO_VALID_CALL" if r.canonical_tool == r.expected_tool or r.parse_status != "ok" else f"other_tool:{r.canonical_tool}"
            p = json.loads(r.params) if isinstance(r.params, str) else {}
            v = p.get(pm[r.prompt_id])
            return f"{v}" if v is not None else "(omitted)"
        bias["method"] = bias.apply(meth, axis=1)
        tab = bias.groupby(["prompt_id", "label"]).method.apply(lambda s: dict(Counter(s))).reset_index()
        tab = bias.groupby(["prompt_id", "label", "method"]).size().unstack(fill_value=0)
        tab.to_csv(OUT / "bbias_method_counts.csv")
        for pid, g in bias.groupby("prompt_id"):
            say(f"  {pid} (schema-text default: {defaults_text[pid]}):")
            for lab, gg in g.groupby("label"):
                say(f"    {lab}: {dict(Counter(gg.method).most_common())}")

    # ---------------- GLMM ----------------
    say("\n== GLMM (lme4::glmer, logistic): outcome ~ mechanism * information + backend + (1 | prompt)")
    gl = main_[["label", "mechanism", "info", "prompt_id", "completed", "tool_correct", "silent_failure"]].copy()
    for c in ["completed", "tool_correct", "silent_failure"]:
        gl[c] = gl[c].astype(int)
    gl.to_csv(OUT / "glmm_input.csv", index=False)
    r_code = f"""
suppressMessages({{library(lme4)}})
d <- read.csv('{OUT / 'glmm_input.csv'}')
d$mechanism <- factor(d$mechanism, levels=c('json','native'))
d$info <- factor(d$info, levels=c('full','bare','minimal'))
d$label <- factor(d$label)
for (y in c('completed','tool_correct','silent_failure')) {{
  cat('\\n### outcome:', y, ' (mean', round(mean(d[[y]]),3), ')\\n')
  sep <- FALSE
  fa <- tryCatch(glmer(as.formula(paste(y, '~ mechanism + info + label + (1|prompt_id)')), data=d, family=binomial,
                 control=glmerControl(optimizer='bobyqa', optCtrl=list(maxfun=2e5))), error=function(e) e)
  if (inherits(fa,'error')) cat('additive model failed:', conditionMessage(fa), '\\n') else {{
    co <- summary(fa)$coefficients; i <- 'mechanismnative'
    cat(sprintf('additive: OR(native vs json) = %.2f [%.2f, %.2f], P = %.3g; singular=%s\\n', exp(co[i,1]),
        exp(co[i,1]-1.96*co[i,2]), exp(co[i,1]+1.96*co[i,2]), co[i,4], isSingular(fa)))
    print(round(co,3)) }}
  fi <- tryCatch(glmer(as.formula(paste(y, '~ mechanism * info + label + (1|prompt_id)')), data=d, family=binomial,
                 control=glmerControl(optimizer='bobyqa', optCtrl=list(maxfun=2e5))), error=function(e) e)
  if (inherits(fi,'error')) cat('interaction model failed:', conditionMessage(fi), '\\n') else {{
    co <- summary(fi)$coefficients; V <- vcov(fi)
    for (lv in c('full','bare','minimal')) {{
      L <- setNames(rep(0, nrow(co)), rownames(co)); L['mechanismnative'] <- 1
      nm <- paste0('mechanismnative:info', lv); if (nm %in% names(L)) L[nm] <- 1
      est <- sum(L*co[,1]); se <- sqrt(as.numeric(t(L) %*% as.matrix(V) %*% L))
      cat(sprintf('interaction: OR(native vs json | info=%s) = %.2f [%.2f, %.2f], P = %.3g\\n', lv, exp(est), exp(est-1.96*se), exp(est+1.96*se), 2*pnorm(-abs(est/se))))
    }}
    cat('singular=', isSingular(fi), '\\n')
    if (any(abs(co[,1]) > 10) || any(co[,2] > 50)) sep <- TRUE }}
  if (inherits(fa,'error') || inherits(fi,'error')) sep <- TRUE
  if (!inherits(fa,'error')) {{ ca <- summary(fa)$coefficients; if (any(abs(ca[,1]) > 10) || any(ca[,2] > 50)) sep <- TRUE }}
  if (sep && length(unique(d[[y]])) > 1) {{
    suppressMessages(library(brglm2))
    cat('separation detected -> bias-reduced GLM (brglm2, prompt as fixed effect), as pre-specified\\n')
    ba <- glm(as.formula(paste(y, '~ mechanism + info + label + prompt_id')), data=d, family=binomial, method='brglmFit')
    cb <- summary(ba)$coefficients; i <- 'mechanismnative'
    cat(sprintf('brglm2 additive: OR(native vs json) = %.2f [%.2f, %.2f], P = %.3g\\n', exp(cb[i,1]),
        exp(cb[i,1]-1.96*cb[i,2]), exp(cb[i,1]+1.96*cb[i,2]), cb[i,4]))
    bi <- glm(as.formula(paste(y, '~ mechanism * info + label + prompt_id')), data=d, family=binomial, method='brglmFit')
    cb <- summary(bi)$coefficients; V <- vcov(bi)
    for (lv in c('full','bare','minimal')) {{
      L <- setNames(rep(0, nrow(cb)), rownames(cb)); L['mechanismnative'] <- 1
      nm <- paste0('mechanismnative:info', lv); if (nm %in% names(L)) L[nm] <- 1
      est <- sum(L*cb[,1]); se <- sqrt(as.numeric(t(L) %*% as.matrix(V[names(L),names(L)]) %*% L))
      cat(sprintf('brglm2 interaction: OR(native vs json | info=%s) = %.2f [%.2f, %.2f], P = %.3g\\n', lv, exp(est), exp(est-1.96*se), exp(est+1.96*se), 2*pnorm(-abs(est/se))))
    }}
  }}
}}
"""
    out = subprocess.run(["Rscript", "-e", r_code], capture_output=True, text=True)
    say(out.stdout.strip())
    if out.returncode != 0 or "failed" in out.stdout:
        say(out.stderr[-2000:])
    (OUT / "expB_summary.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Experiment A: derive parameter grids from the Pydantic schema of identify_spatial_domains.

The generator reads ``SpatialDomainParameters.model_fields`` (type annotation,
Literal choices, ge/gt/le/lt bounds, default) and applies fixed discretization
rules. Nothing here is tuned on data or labels.

Discretization rules (frozen in PREREGISTRATION.md):
  R1  Literal[...] and bool             -> every value.
  R2  float with both bounds, range <= 1 (weights, fractions, thresholds)
                                         -> interior points lo + W*{.1,.3,.5,.7,.9}, plus the default.
  R3  int with <= 5 legal values         -> every legal value.
  R4  n_domains (the domain-count target) -> unit steps default-3 .. default+3 within bounds.
  R5  other int counts (wide range)      -> default * {0.5, 1, 2}, rounded, clipped to bounds.
  R6  float bounded below only, or unbounded (scales such as resolutions)
                                         -> default * {0.25, 0.5, 1, 2, 4}.
  R7  Optional[...] = None: enumerated with the rule of its type around the
      effective default, read from the field description ("None uses 0.3",
      "k=15"); the effective-default rung is kept as None so the schema default
      is a grid point. When the description says the default is derived from
      the data ("None derives it from the spot spacing", stagate_rad_cutoff in
      1.5.5), rungs are multiples of that data-derived default ("auto*f"),
      resolved per input with ChatSpatial's own derivation.
  (Pilot grid on the pre-release branch: effective defaults were read from the
  code and stagate_rad_cutoff used a declared data-scale ladder; see
  PREREGISTRATION.md.)
Held fixed (not tuning parameters or not applicable): random seeds, *_use_gpu,
timeout, spagcn_compat (fixed to "spagcn-1.2.7"), use_highly_variable
(feature-selection scope, fixed by preprocessing), graphst_n_clusters (alias
of n_domains), parameters a method does not consume, and parameters inert on the
data (spagcn_s / spagcn_b / spagcn_use_histology when no histology image).
Post-processing parameters (refine_domains, refinement_threshold) are
enumerated without extra runs: the unrefined labels of a call are refined with
ChatSpatial's own _refine_spatial_domains at each threshold.
Budget rule: if the factorial of a method's run-level parameters exceeds
MAX_CALLS, a deterministic random subset of MAX_CALLS is drawn (numpy seed 0),
always containing the schema default.
"""
from __future__ import annotations

import itertools
import json
import sys
import types
import typing
from typing import Any, Literal, get_args, get_origin

import numpy as np

MAX_CALLS = 150
SEED_FIELDS_SUFFIX = ("_random_seed",)
HELD = {"timeout", "use_highly_variable", "spagcn_compat", "graphst_n_clusters",
        "stagate_use_gpu", "graphst_use_gpu"}
POSTPROC = ("refine_domains", "refinement_threshold")
# Effective defaults of Optional fields, read from tools/spatial_domains.py
# (the schema itself does not carry them; reported as a schema gap).
EFFECTIVE_DEFAULTS = {"cluster_n_neighbors": 15, "cluster_spatial_weight": 0.3,
                      "stagate_rad_cutoff": 50.0}
UNITLESS_DISTANCE = {"stagate_rad_cutoff"}
DATA_SCALE_LADDER = (1.1, 2.0, 3.0)  # x median nearest-neighbour distance
INERT_WITHOUT_IMAGE = {"spagcn_s", "spagcn_b", "spagcn_use_histology"}
METHODS = ("spagcn", "leiden", "louvain", "stagate", "graphst", "banksy")
# Generic (unprefixed) fields consumed by each backend, read from
# tools/spatial_domains.py. n_domains is ignored by resolution-driven backends
# (the package's own _domain_count_control says so).
GENERIC = {
    "spagcn": ["n_domains"],
    "leiden": ["resolution", "cluster_n_neighbors", "cluster_spatial_weight"],
    "louvain": ["resolution", "cluster_n_neighbors", "cluster_spatial_weight"],
    "stagate": ["n_domains"],
    "graphst": ["n_domains"],
    "banksy": [],
}


CONDITIONAL = {"graphst_radius": ("graphst_refinement", True)}


def defaults_of(name):
    return _schema().model_fields[name].default


def _schema():
    from chatspatial.models.data import SpatialDomainParameters
    return SpatialDomainParameters


def _bounds(field) -> dict:
    out = {}
    for m in field.metadata:
        for k in ("ge", "gt", "le", "lt"):
            v = getattr(m, k, None)
            if v is not None:
                out[k] = v
    return out


def _base_type(ann):
    """Strip Optional / Annotated; return (base, choices or None, optional?)."""
    optional = False
    origin = get_origin(ann)
    if origin in (typing.Union, types.UnionType):
        args = [a for a in get_args(ann) if a is not type(None)]
        optional = len(args) < len(get_args(ann))
        ann = args[0] if len(args) == 1 else ann
        origin = get_origin(ann)
    if origin is typing.Annotated:
        ann = get_args(ann)[0]
        origin = get_origin(ann)
    if origin is Literal:
        return "literal", list(get_args(ann)), optional
    if ann is bool:
        return "bool", [False, True], optional
    if ann is int:
        return "int", None, optional
    if ann is float:
        return "float", None, optional
    return "other", None, optional


def effective_default(name: str, field):
    """Schema default, or for Optional fields the effective default stated in the description."""
    import re
    if field.default is not None:
        return field.default
    desc = field.description or ""
    m = re.search(r"None uses ([0-9]+(?:\.[0-9]+)?)", desc)
    if m:
        return float(m.group(1)) if "." in m.group(1) else int(m.group(1))
    m = re.search(r"\bk=([0-9]+)", desc)
    if m:
        return int(m.group(1))
    if re.search(r"None derives it from", desc):
        return "auto"
    return EFFECTIVE_DEFAULTS.get(name)  # pre-release schema fallback


def field_values(name: str, field) -> tuple[list, str]:
    """Discretize one schema field. Returns (values, rule id)."""
    kind, choices, optional = _base_type(field.annotation)
    b = _bounds(field)
    default = field.default
    eff = effective_default(name, field)
    lo = b.get("ge", b.get("gt"))
    hi = b.get("le", b.get("lt"))

    def clip(v):
        if "ge" in b and v < b["ge"]:
            return None
        if "gt" in b and v <= b["gt"]:
            return None
        if "le" in b and v > b["le"]:
            return None
        if "lt" in b and v >= b["lt"]:
            return None
        return v

    if kind in ("literal", "bool"):
        return list(choices), "R1"
    if eff == "auto":  # data-derived effective default (R6/R5 around it)
        facs = (0.25, 0.5, 2, 4) if kind == "float" else (0.5, 2)
        return [None] + [f"auto*{f}" for f in facs], ("R6" if kind == "float" else "R5") + "+R7-data"
    if eff is None:
        return [default], "held (no effective default in schema)"
    if kind == "float" and lo is not None and hi is not None and hi - lo <= 1:
        W = hi - lo
        vals = sorted({round(lo + W * q, 6) for q in (0.1, 0.3, 0.5, 0.7, 0.9)} | {eff})
        rule = "R2"
    elif kind == "int" and lo is not None and hi is not None and (hi - lo) <= 4:
        start = b["ge"] if "ge" in b else b["gt"] + 1
        stop = b["le"] if "le" in b else b["lt"] - 1
        vals, rule = list(range(start, stop + 1)), "R3"
    elif name == "n_domains":
        vals = [v for v in range(eff - 3, eff + 4) if clip(v) is not None]
        rule = "R4"
    elif kind == "int":
        vals = sorted({c for c in (int(round(eff * f)) for f in (0.5, 1, 2))
                       if clip(c) is not None})
        rule = "R5"
    elif kind == "float":
        vals = sorted({c for c in (round(eff * f, 6) for f in (0.25, 0.5, 1, 2, 4))
                       if clip(c) is not None})
        rule = "R6"
    else:
        return [default], "held"
    if default is None:  # R7: the effective-default rung stays None (schema default)
        vals = [None if v == eff else v for v in vals]
        rule += "+R7"
    return vals, rule


def generic_applies(name: str, field, method: str):
    """Applicability of an unprefixed field to `method`, as stated in its description (or None)."""
    import re
    desc = field.description or ""
    m = re.search(r"\(([a-z/]+) only\)", desc)
    if m:
        return method in m.group(1).split("/")
    m = re.search(r"Used by ([^.]*)\.", desc)
    if m:
        used = re.findall(r"[a-z]+", m.group(1).split("(")[0] + " " + m.group(1))
        ignored = re.search(r"Ignored by ([^.]*)\.", desc)
        if ignored and method in re.findall(r"[a-z]+", ignored.group(1)):
            return False
        return method in used
    return None


def method_fields(method: str, has_image: bool = False) -> list[str]:
    S = _schema()
    names = []
    for name in S.model_fields:
        if name == "method" or name in HELD or name in POSTPROC:
            continue
        if name.endswith(SEED_FIELDS_SUFFIX):
            continue
        stated = None if name.startswith(tuple(m + "_" for m in METHODS + ("aestetik",))) \
            else generic_applies(name, S.model_fields[name], method)
        if stated is not None and stated != (name in GENERIC[method]):
            print(f"NOTE: schema says {name} applies to {method}={stated}; code map says {name in GENERIC[method]}")
        if name.startswith(method + "_") or (stated if stated is not None else name in GENERIC[method]):
            if not has_image and name in INERT_WITHOUT_IMAGE:
                continue
            names.append(name)
    return names


def method_grid(method: str, has_image: bool = False) -> dict:
    """Run-level configurations (one tool call each) for one method."""
    S = _schema()
    fields = method_fields(method, has_image)
    axes, rules = {}, {}
    for f in fields:
        axes[f], rules[f] = field_values(f, S.model_fields[f])
    defaults = {f: S.model_fields[f].default for f in fields}
    full = [dict(zip(axes, combo)) for combo in itertools.product(*axes.values())]
    # Conditional parameters (read from the code; the schema does not state them):
    # graphst_radius is used only when graphst_refinement is True.
    seen, dedup = set(), []
    for c in full:
        for child, (parent, active) in CONDITIONAL.items():
            if child in c and c.get(parent) != active:
                c[child] = defaults_of(child)
        key = json.dumps(c, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            dedup.append(c)
    full = dedup
    n_full = len(full)
    if n_full > MAX_CALLS:
        rng = np.random.default_rng(0)
        idx = sorted(rng.choice(n_full, size=MAX_CALLS, replace=False).tolist())
        chosen = [full[i] for i in idx]
        if defaults not in chosen:
            chosen[-1] = defaults
        configs = chosen
    else:
        configs = full
    # default first, stable order afterwards
    configs = sorted(configs, key=lambda c: (c != defaults, json.dumps(c, sort_keys=True, default=str)))
    post = postproc_variants()
    return {"method": method, "axes": axes, "rules": rules, "defaults": defaults,
            "n_factorial": n_full, "configs": configs,
            "n_calls": len(configs), "n_post_variants": len(post),
            "n_configurations": len(configs) * len(post)}


def postproc_variants() -> list[dict]:
    S = _schema()
    thr, _ = field_values("refinement_threshold", S.model_fields["refinement_threshold"])
    return [{"refine_domains": False, "refinement_threshold": S.model_fields["refinement_threshold"].default}] + \
           [{"refine_domains": True, "refinement_threshold": t} for t in thr]


def default_postproc() -> dict:
    S = _schema()
    return {"refine_domains": S.model_fields["refine_domains"].default,
            "refinement_threshold": S.model_fields["refinement_threshold"].default}


def config_id(method: str, cfg: dict) -> str:
    items = ",".join(f"{k}={cfg[k]}" for k in sorted(cfg))
    return f"{method}|{items}"


def resolve(cfg: dict, coords) -> dict:
    """Turn data-derived tokens into numbers for this input.

    "auto*f": f x ChatSpatial's own default derivation for that field
    (stagate_rad_cutoff: tools.spatial_domains._stagate_auto_radius).
    "nn*f" (pilot grid): f x median nearest-neighbour distance.
    """
    import numpy as np
    out = {}
    for k, v in cfg.items():
        if isinstance(v, str) and v.startswith("auto*"):
            if k != "stagate_rad_cutoff":
                raise ValueError(f"no data-derived default known for {k}")
            from chatspatial.tools.spatial_domains import _stagate_auto_radius
            out[k] = round(float(v[5:]) * _stagate_auto_radius(np.asarray(coords)), 4)
        elif isinstance(v, str) and v.startswith("nn*"):
            from sklearn.neighbors import NearestNeighbors
            d, _ = NearestNeighbors(n_neighbors=2).fit(coords).kneighbors(coords)
            out[k] = round(float(v[3:]) * float(np.median(d[:, 1])), 4)
        else:
            out[k] = v
    return out


def main():
    rows = []
    total_calls = total_cfg = 0
    for m in METHODS:
        g = method_grid(m)
        total_calls += g["n_calls"]
        total_cfg += g["n_configurations"]
        print(f"\n== {m}: {g['n_calls']} calls (factorial {g['n_factorial']}), "
              f"{g['n_configurations']} configurations incl. {g['n_post_variants']} post-processing variants")
        for f, vals in g["axes"].items():
            print(f"   {f:28s} {g['rules'][f]:14s} {vals}   (default {g['defaults'][f]})")
        rows.append(g)
    print(f"\nTOTAL: {total_calls} tool calls / section (one replicate), "
          f"{total_cfg} configurations / section")
    if len(sys.argv) > 1:
        with open(sys.argv[1], "w") as fh:
            json.dump(rows, fh, indent=1, default=str)


if __name__ == "__main__":
    main()

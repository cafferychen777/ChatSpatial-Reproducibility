#!/usr/bin/env python3
"""Remove result rows affected by the gmm_clustering bug (fixed in ChatSpatial 1.5.9).

Affected: every STAGATE unit; every GraphST unit that clusters with mclust (the
sklearn EEE substitute), i.e. graphst_clustering_method="mclust" and the expert
strategy. In noise stages, GraphST rule/oracle units are also removed because the
re-run rule selection may pick different configurations.
Usage: purge_affected.py STAGE [--apply]   (backs up the CSVs to one tar.gz first)
"""
import csv
import io
import sys
import tarfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def affected(d, noise):
    if d["method"] == "stagate":
        return True
    if d["method"] == "graphst":
        if d["strategy"] == "expert" or '"graphst_clustering_method": "mclust"' in d["config"]:
            return True
        if noise and d["strategy"] in ("rule", "oracle"):
            return True
    return False


def main():
    stage, apply = sys.argv[1], "--apply" in sys.argv
    noise = stage.startswith("noise")
    base = C.ROOT / "results" / stage
    files = [f for f in base.rglob("task_*.csv") if f.parent.name in ("stagate", "graphst")]
    if apply:
        bdir = C.ROOT / "backups" / "gmm_fix_2026-10-09"
        bdir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(bdir / f"{stage}_stagate_graphst_csv_{time.strftime('%H%M%S')}.tar.gz", "w:gz") as tf:
            for f in files:
                tf.add(f, arcname=str(f.relative_to(C.ROOT)))
    n_drop = n_keep = 0
    for f in files:
        rows = list(csv.reader(io.StringIO(f.read_text())))
        header, keep = rows[0], []
        for r in rows[1:]:
            d = dict(zip(header, r))
            if len(r) == len(header) and affected(d, noise):
                n_drop += 1
            else:
                keep.append(r)
        n_keep += len(keep)
        if apply:
            buf = io.StringIO()
            w = csv.writer(buf, lineterminator="\n")
            w.writerow(header)
            w.writerows(keep)
            tmp = f.with_suffix(".csv.tmp")
            tmp.write_text(buf.getvalue())
            tmp.replace(f)
    print(f"{stage}: {'removed' if apply else 'would remove'} {n_drop} rows; kept {n_keep} "
          f"STAGATE/GraphST rows in {len(files)} files")


if __name__ == "__main__":
    main()

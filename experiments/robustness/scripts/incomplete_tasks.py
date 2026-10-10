#!/usr/bin/env python3
"""Find tasks of a stage with missing units, after purging rows lost to I/O failures.

Usage: incomplete_tasks.py STAGE [--purge]
Rows whose error is an I/O failure (disk quota, no space, write errors) are not
results; --purge removes them (and malformed CSV lines) so the resumed task reruns
those units. Prints the comma-separated task indices that still lack units.
"""
import csv
import io
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

IOERR = re.compile(r"quota|no space|errno 28|errno 122|OSError|BlockingIOError|"
                   r"Read-only file system|Input/output error|unable to (open|write|create)|allow_pickle|No such file or directory", re.I)


def main():
    stage = sys.argv[1]
    purge = "--purge" in sys.argv
    extra = None  # --rerun-pattern REGEX: also drop error rows matching REGEX (fixed backend bugs)
    if "--rerun-pattern" in sys.argv:
        extra = re.compile(sys.argv[sys.argv.index("--rerun-pattern") + 1])
    tdir = C.ROOT / "tasks" / stage
    todo, purged = [], 0
    for tf in sorted(tdir.glob("task_*.json")):
        T = json.loads(tf.read_text())
        out = (C.ROOT / "results" / stage / T["dataset"] / T["section"] / T["condition"] /
               T["method"] / f"{tf.stem}.csv")
        need = {(c["call_id"], r) for c in T["calls"] for r in T["reps"]}
        have = set()
        if out.exists():
            rows = list(csv.reader(io.StringIO(out.read_text())))
            header, good, bad = (rows[0] if rows else []), [], 0
            # a replicate that could not find its full-data labels invalidates that call's full row too
            lost = {dict(zip(header, r))["call_id"] for r in rows[1:]
                    if len(r) == len(header) and "No such file or directory" in dict(zip(header, r))["error"]}
            for row in rows[1:]:
                if len(row) != len(header):
                    bad += 1
                    continue
                d = dict(zip(header, row))
                if (d["status"] != "ok" and (IOERR.search(d["error"]) or (extra and extra.search(d["error"])))) \
                        or d["call_id"] in lost:
                    bad += 1
                    continue
                good.append(row)
                have.add((d["call_id"], d["rep"]))
            if bad and purge:
                buf = io.StringIO()
                w = csv.writer(buf, lineterminator="\n")
                w.writerow(header)
                w.writerows(good)
                tmp = out.with_suffix(".csv.tmp")
                tmp.write_text(buf.getvalue())
                tmp.replace(out)
                purged += bad
        if need - have:
            todo.append(T["index"] if "index" in T else int(tf.stem.split("_")[1]))
    print(f"# {stage}: purged {purged} rows; {len(todo)} incomplete tasks", file=sys.stderr)
    print(",".join(map(str, todo)))


if __name__ == "__main__":
    main()

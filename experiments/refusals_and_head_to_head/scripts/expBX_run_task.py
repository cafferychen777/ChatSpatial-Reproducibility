#!/usr/bin/env python3
"""Run one experiment B-X trial (called by the SLURM array; standard library only).

Usage: python3 expBX_run_task.py TASKFILE INDEX
  TASKFILE = jsonl of tasks (expBX_make_tasks.py); INDEX = 0-based line (SLURM_ARRAY_TASK_ID,
  optionally offset). Resumable: a task whose status.json says finished is skipped.
Layout: <run_dir>/ harness files (status.json, result.json, transcript or api_log, stdout.log);
        <run_dir>/out/ the competitor agent's workspace (the output location named in its prompt).
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import expBX_common as C  # noqa: E402

S = str(C.WORK)
# ChatSpatial env: 1.5.6 for the probes and the first head-to-head run, 1.5.9 for the head-to-head re-run
# (EXPBX_CS_ENV=$S/conda_envs/chatspatial_bx159); rules section 6
V155 = os.environ.get("EXPBX_CS_ENV", f"{S}/conda_envs/chatspatial_bx156")


# Open-weight arm (GPU host): the CPU-cluster trees are mirrored under EXPBX_HOST_PREFIX and bound to their original
# absolute paths inside the container, so every path a system sees is unchanged.
HOSTP = os.environ.get("EXPBX_HOST_PREFIX", "")
APPTAINER = os.environ.get("EXPBX_APPTAINER", "apptainer")
IMAGE = f"{C.ROOT}/images/jammy-scm.sif"  # buildpack-deps:jammy-scm (Ubuntu 22.04 + git, curl, wget)
# uv-managed CPython 3.11.14 that the competitor venvs were built with
UV_PY = os.environ.get("EXPBX_UV_PY", "/path/to/uv/python/cpython-3.11.14-linux-x86_64-gnu")
CONTAINER = os.environ.get("EXPBX_CONTAINER", "apptainer")  # apptainer (CPU cluster) | docker (GPU host, open-weight arm)
DOCKER_IMAGE = os.environ.get("EXPBX_DOCKER_IMAGE", "buildpack-deps:jammy-scm")


def env_for(system: str, tmp: Path, variant: str | None) -> tuple[list[str], dict, list[str]]:
    """(interpreter argv, environment inside the container, read-only binds)."""
    threads = os.environ.get("SLURM_CPUS_PER_TASK", "4")
    env = dict(PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1", MPLBACKEND="Agg", TMPDIR=str(tmp),
               OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads, OPENBLAS_NUM_THREADS=threads,
               NUMBA_NUM_THREADS=threads, NUMBA_CACHE_DIR=str(tmp / "numba"), EXPBX_ROOT=str(C.ROOT),
               LANG="C.UTF-8", LC_ALL="C.UTF-8")
    ro = [str(C.SCRIPTS)]
    if system in ("chatspatial", "chatspatial_single"):
        pyl = [f"{C.ROOT}/pylib", f"{S}/expB/compat", str(C.SCRIPTS)]
        env.update(PATH=f"{V155}/bin:/usr/bin:/bin",
                   R_LIBS_USER=f"{S}/expB/rlib_v155:{S}/expB/no_user_rlib", R_LIBS="",
                   R_ENVIRON_USER="/dev/null", R_PROFILE_USER="/dev/null", R_MAKEVARS_USER="/dev/null",
                   PYTHONPATH=os.pathsep.join(pyl))
        ro += [V155, f"{C.ROOT}/pylib", f"{S}/expB/compat", f"{S}/expB/rlib_v155", f"{S}/expB/no_user_rlib"]
        return [f"{V155}/bin/python", "-W", "ignore"], env, ro
    venv = C.COMPETITORS / ("STAgent" if system == "stagent" else "SpatialAgent") / ".venv"
    env.update(PATH=f"{venv}/bin:/usr/bin:/bin", VIRTUAL_ENV=str(venv),
               GOOGLE_API_KEY="dummy-not-used", OPENAI_API_KEY="dummy-not-used",
               PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python")
    ro += [UV_PY]
    return [f"{venv}/bin/python"], env, ro


def _count(p: Path, cap: int = 100000) -> int:
    n = 0
    for _, ds, fs in os.walk(p):
        n += len(ds) + len(fs)
        if n > cap:
            break
    return n


def copy_back(src: Path, dst: Path, log: list, top: bool = True) -> None:
    """Copy a finished run's node-local directory to scratch (rules section 6, 2026-10-09 file-count quota fix).
    Regular files are copied; a subdirectory holding > 1000 entries (a package install tree) is not copied,
    except its regular files <= 5 MB at depth <= 2, which go to <name>__kept/. Files > 4 GB are skipped."""
    import shutil
    for item in sorted(src.iterdir()):
        target = dst / item.name
        try:
            if item.is_symlink():
                continue
            if item.is_file():
                if item.stat().st_size > 4e9:
                    log.append({"skipped_large_file": str(item.name), "bytes": item.stat().st_size})
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, target)
            elif item.is_dir():
                n = _count(item)
                if n > 1000 and not (top and item.name == "out"):  # the requested-output dir is always walked
                    log.append({"install_tree_not_copied": item.name,
                                "entries": n})
                    for dp, ds, fs in os.walk(item):
                        rel = Path(dp).relative_to(item)
                        if len(rel.parts) > 2:
                            ds[:] = []
                            continue
                        for fn in fs:
                            f = Path(dp) / fn
                            if f.is_file() and not f.is_symlink() and f.stat().st_size <= 5_000_000:
                                d = dst / (item.name + "__kept") / rel / fn
                                d.parent.mkdir(parents=True, exist_ok=True)
                                shutil.copy2(f, d)
                else:
                    target.mkdir(parents=True, exist_ok=True)
                    copy_back(item, target, log, top=False)
        except OSError as e:
            log.append({"copy_error": item.name, "error": str(e)[:200]})


def rm_tree(p: Path) -> None:
    import shutil
    import stat as st_
    for dp, ds, fs in os.walk(p):
        for x in ds:
            try:
                os.chmod(os.path.join(dp, x), 0o700)
            except OSError:
                pass
    shutil.rmtree(p, ignore_errors=True)


def input_dir(task: dict) -> Path:
    return C.dlpfc_input(task["unit"]).parent if task["part"] == "h2h" else C.probe_dir(task["unit"])


def command(task: dict, rd: Path) -> list[str]:
    s, part, unit = task["system"].split("@")[0], task["part"], task["unit"]
    out = rd / "out"
    if s == "chatspatial_single":
        return [str(C.SCRIPTS / "expBX_single_call.py"), "--sid", unit, "--out_dir", str(rd)]
    if s == "chatspatial":
        if part == "h2h":
            prompt = C.dlpfc_agent_prompt(unit, None)
            return [str(C.SCRIPTS / "expBX_chatspatial_agent.py"), "--prompt", prompt, "--out_dir", str(rd), "--h2h"]
        return [str(C.SCRIPTS / "expBX_chatspatial_agent.py"), "--prompt", C.probe_agent_prompt(unit, None),
                "--out_dir", str(rd)]
    driver = C.SCRIPTS / f"expBX_driver_{s}.py"
    if part == "h2h":
        data_path = C.dlpfc_input(unit)
        prompt = C.dlpfc_agent_prompt(unit, str(out / "result.h5ad"))
    else:
        p = C.PROBE_BY_ID[unit]
        data_path = C.probe_dir(unit) / p["files"]["spatial"]
        prompt = C.probe_agent_prompt(unit, str(out))
    return [str(driver), "--data_path", str(data_path), "--output_dir", str(out), "--prompt", prompt,
            "--log_dir", str(rd)]


def main():
    taskfile, idx = sys.argv[1], int(sys.argv[2])
    tasks = [json.loads(l) for l in open(taskfile) if l.strip()]
    if idx >= len(tasks):
        print("index beyond task list", idx, len(tasks))
        return
    task = tasks[idx]
    rd = C.run_dir(task)
    H = lambda p: Path(HOSTP + str(p))  # noqa: E731  host-side path
    st = H(rd) / "status.json"
    if st.exists() and json.loads(st.read_text()).get("finished"):
        print("already finished", task["task_id"])
        return
    H(rd).mkdir(parents=True, exist_ok=True)
    tmp = C.ROOT / "tmp" / task["task_id"]  # container path of the private HOME/TMPDIR
    # node-local working copies (rules section 6, 2026-10-09): the run directory and HOME live on the node's /tmp
    # and are bound to the same absolute paths; only results are copied back to scratch afterwards.
    LOCAL = Path(os.environ.get("EXPBX_LOCAL", "/tmp")) / f"expBX_{os.environ.get('SLURM_JOB_ID', 'x')}_{task['task_id']}"
    if LOCAL.exists():
        rm_tree(LOCAL)
    L_rd, L_home = LOCAL / "run", LOCAL / "home"
    (L_rd / "out").mkdir(parents=True, exist_ok=True)
    L_home.mkdir(parents=True, exist_ok=True)
    system, _, variant = task["system"].partition("@")
    prefix, cenv, ro = env_for(system, tmp, variant or None)
    rw = [str(rd)]  # tmp is the container HOME (--home), also read-write
    if system in ("stagent", "spatialagent"):
        rw.append(str(C.COMPETITORS / ("STAgent" if system == "stagent" else "SpatialAgent")))
    ro.append(str(input_dir(task)))
    # Isolation (rules section 6, 2026-10-09): the run sees only its own input directory, the code and
    # environment it needs (read-only), and its own run/tmp directories (read-write). Same absolute paths.
    binds = [f"{HOSTP}{p}:{p}:ro" for p in ro] + [f"{HOSTP}{p}:{p}" for p in rw[1:]] + [f"{L_rd}:{rd}"]
    cenv.update({"EXPBX_BACKEND": C.BACKEND, "EXPBX_MODEL": C.MODEL, "EXPBX_BASE_URL": C.BASE_URL})
    cwd = rd / "out" if system in ("stagent", "spatialagent") else rd
    if CONTAINER == "docker":
        # GPU host of the open-weight arm (rules section 6): Docker with read-only binds; host network for the local vLLM endpoint
        cname = f"expbx_{os.getpid()}_{int(time.time())}"
        cmd = ["docker", "run", "--rm", "--name", cname, "--network", "host", "--user", f"{os.getuid()}:{os.getgid()}",
               "--workdir", str(cwd), "-v", f"{L_home}:{tmp}"]
        for b in binds:
            src = b.split(":")[0]
            if Path(src).exists():
                cmd += ["-v", b]
        for k, v in {**cenv, "HOME": str(tmp)}.items():
            cmd += ["-e", f"{k}={v}"]
        # optional extra docker flags (Qwen head-to-head on the GPU host: --oom-score-adj so that under memory pressure the
        # kernel kills a run container, not the shared vLLM server)
        cmd += os.environ.get("EXPBX_DOCKER_EXTRA", "").split()
        cmd += [DOCKER_IMAGE] + prefix + command(task, rd)
    else:
        cname = None
        cmd = [APPTAINER, "exec", "--containall", "--no-mount", "cwd", "--home", f"{L_home}:{tmp}", "--pwd", str(cwd)]
        for b in binds:
            cmd += ["--bind", b]
        for k, v in cenv.items():
            cmd += ["--env", f"{k}={v}"]
        cmd += [HOSTP + IMAGE] + prefix + command(task, rd)
    env = {"PATH": "/usr/bin:/bin", "HOME": os.environ.get("HOME", "")}
    if C.BACKEND == "anthropic":
        env["APPTAINERENV_ANTHROPIC_API_KEY"] = C.anthropic_key()  # key passed via environment, never on argv
    t0 = time.time()
    (L_home / "numba").mkdir(exist_ok=True)
    status = {"task": task, "host": os.uname().nodename, "isolation": "docker (ro binds)" if CONTAINER == "docker" else "apptainer --containall", "backend": C.BACKEND, "model": C.MODEL, "slurm_job": os.environ.get("SLURM_JOB_ID"),
              "array_task": os.environ.get("SLURM_ARRAY_TASK_ID"), "start": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "timeout_s": C.RUN_TIMEOUT_S, "finished": False}
    st.write_text(json.dumps(status, indent=1))
    print(f"{status['start']} {task['task_id']} on {status['host']}", flush=True)
    with open(H(rd) / "stdout.log", "w") as log:
        p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(L_rd), start_new_session=True)
        try:
            rc = p.wait(timeout=C.RUN_TIMEOUT_S)
            timed_out = False
        except subprocess.TimeoutExpired:
            if cname:
                subprocess.run(["docker", "kill", cname], capture_output=True)
            os.killpg(p.pid, signal.SIGTERM)
            time.sleep(10)
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            rc, timed_out = None, True
    cb_log = []
    copy_back(L_rd, H(rd), cb_log)
    rm_tree(LOCAL)
    status.update(finished=True, returncode=rc, timed_out=timed_out, wall_time=time.time() - t0,
                  end=time.strftime("%Y-%m-%dT%H:%M:%S"), result_json=(H(rd) / "result.json").exists(),
                  copy_back=cb_log, workdir="node-local")
    st.write_text(json.dumps(status, indent=1))
    print(f"done rc={rc} timeout={timed_out} {status['wall_time']:.0f}s", flush=True)


if __name__ == "__main__":
    main()

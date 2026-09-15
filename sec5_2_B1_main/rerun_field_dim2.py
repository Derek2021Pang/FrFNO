# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_PKG_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _PKG_ROOT not in _sys.path: _sys.path.insert(0, _PKG_ROOT)
"""
Rerun all core experiments with field_dim=2 (default in frfno_core.py).
Runs sequentially, saves results incrementally, supports checkpoint resume.
Usage: python rerun_field_dim2.py
"""
import os, sys, time, subprocess

ROOT = _PKG_ROOT
PYTHON = sys.executable
WORKDIR = ROOT
LOGFILE = os.path.join(WORKDIR, "rerun_field_dim2.log")
PROGRESS_FILE = os.path.join(WORKDIR, "rerun_field_dim2_progress.txt")

# List of (script_name, description, result_file, estimated_minutes)
EXPERIMENTS = [
    ("b1_burgers.py", "B1: Main nonlinear test (7 models)", "b1_burgers_result.txt", 25),
    ("b2b_longT.py", "B2: Long integration window T=0.06", "b2b_longT_result.txt", 22),
    ("b3_pino.py", "B3: Space-time PINO comparison", "b3_result.txt", 19),
    ("b4_integer.py", "B4: Integer-order limit", "b4_result.txt", 5),
    ("theory_verify.py", "Theory verification: Exp1-5 (falsification)", "theory_verify_result.txt", 16),
    ("riesz_periodic_check.py", "Equation breadth: Periodic Riesz", "riesz_periodic_result.txt", 5),
    ("reaction_diffusion_check.py", "Equation breadth: Reaction-diffusion (3 mechanisms)", "reaction_diffusion_result.txt", 15),
    ("system_burgers_check.py", "Equation breadth: Two-component vector Burgers", "system_burgers_result.txt", 7),
    ("system_coupled_check.py", "Equation breadth: Non-symmetric coupled system", "system_coupled_result.txt", 11),
    ("ns_vorticity_check.py", "Equation breadth: 2D incompressible NS vorticity", "ns_vorticity_result.txt", 5),
    ("ns3d_vorticity_check.py", "Equation breadth: 3D vector vorticity NS", "ns3d_result.txt", 2),
]

def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOGFILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def save_progress(completed, elapsed_total):
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        f.write(f"completed={completed}\n")
        f.write(f"elapsed_total={elapsed_total:.2f}\n")
        f.write(f"last_update={time.strftime('%Y-%m-%d %H:%M:%S')}\n")

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
            progress = {}
            for line in lines:
                if "=" in line:
                    k, v = line.strip().split("=", 1)
                    progress[k] = v
            return int(progress.get("completed", 0)), float(progress.get("elapsed_total", 0))
    return 0, 0.0

def main():
    total_est = sum(e[3] for e in EXPERIMENTS)
    log("=" * 70)
    log(f"Starting rerun of {len(EXPERIMENTS)} experiments with field_dim=2")
    log(f"Estimated total time: {total_est} minutes ({total_est/60:.1f} hours)")
    log(f"Actual expected: ~{int(total_est*0.55)} minutes (GPU speedup)")
    log("=" * 70)

    # Load progress for checkpoint resume
    start_idx, elapsed_total = load_progress()
    if start_idx > 0:
        log(f"Resuming from checkpoint: {start_idx}/{len(EXPERIMENTS)} completed, {elapsed_total:.1f} min elapsed")

    completed = start_idx

    for i in range(start_idx, len(EXPERIMENTS)):
        script, desc, result_file, est_min = EXPERIMENTS[i]

        log(f"\n--- Experiment {i+1}/{len(EXPERIMENTS)}: {desc} ---")
        log(f"Script: {script}")
        log(f"Estimated time: {est_min} minutes")

        # Check if result already exists (checkpoint)
        result_path = os.path.join(WORKDIR, result_file)
        if os.path.exists(result_path):
            mtime = os.path.getmtime(result_path)
            mtime_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))
            log(f"Result file already exists ({mtime_str}), checking if fresh...")
            # If result file is older than 1 hour, rerun; otherwise skip
            if time.time() - mtime < 3600:
                log(f"Result is fresh (<1 hour old), SKIPPING this experiment")
                completed += 1
                save_progress(completed, elapsed_total)
                continue
            else:
                log(f"Result is stale (>1 hour old), RERUNNING")

        t0 = time.time()
        try:
            result = subprocess.run(
                [PYTHON, script],
                cwd=WORKDIR,
                capture_output=True,
                text=True,
                timeout=est_min * 180,  # 3x safety margin
                encoding="utf-8",
                errors="replace"
            )
            elapsed = (time.time() - t0) / 60
            elapsed_total += elapsed
            completed += 1

            if result.returncode == 0:
                log(f"SUCCESS in {elapsed:.1f} minutes")
            else:
                log(f"FAILED (returncode={result.returncode}) in {elapsed:.1f} minutes")
                log(f"STDERR (last 500 chars): {result.stderr[-500:]}")

            # Save stdout tail
            stdout_tail = result.stdout[-1000:] if result.stdout else ""
            log(f"STDOUT (last 1000 chars): {stdout_tail}")

        except subprocess.TimeoutExpired:
            elapsed = (time.time() - t0) / 60
            elapsed_total += elapsed
            log(f"TIMEOUT after {elapsed:.1f} minutes (limit={est_min*3} min)")

        # Save checkpoint after each experiment
        save_progress(completed, elapsed_total)

        remaining_est = total_est - elapsed_total
        log(f"Progress: {completed}/{len(EXPERIMENTS)} completed, "
            f"elapsed={elapsed_total:.1f} min, estimated remaining={remaining_est:.1f} min")

    log(f"\n{'=' * 70}")
    log(f"All experiments finished: {completed}/{len(EXPERIMENTS)} completed")
    log(f"Total elapsed time: {elapsed_total:.1f} minutes ({elapsed_total/60:.1f} hours)")
    log(f"{'=' * 70}")

if __name__ == "__main__":
    main()

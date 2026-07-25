"""
Written by the notebook's Section 6 cell - the parallel sweep runner.
Materialized to a real file for the same reason as sim_core_v2.py: so
ProcessPoolExecutor can import _sweep_worker from every platform's worker
processes, not just Linux/fork.
"""
import pickle
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import sim_core_v2

MONTH_LEN = 30.44      # days/month (~365/12)
N_MONTHS = 36           # 36-month observation window
DEFAULT_T = 365 * 3

def _sweep_worker(args):
    p_h, p_e, p_u, sim_id, size_pop, T, seed, record_timeseries, fix_known_bugs = args
    return sim_core_v2.simulate_single_run(p_h, p_e, p_u, size_pop=size_pop, T=T, sim_id=sim_id, seed=seed,
                                           record_timeseries=record_timeseries, fix_known_bugs=fix_known_bugs)


def run_sweep(P_h, P_e, P_u, numsims=5, size_pop=10_000, T=DEFAULT_T,
              record_timeseries=True, fix_known_bugs=False, n_workers=None,
              base_seed=None, progress=True):
    """base_seed=None (default) reproduces the original's fully-random behavior.
    Pass an int for reproducible sweeps (useful while debugging/validating)."""
    jobs = []
    for sim in range(numsims):
        for p_h in P_h:
            for p_e in P_e:
                for p_u in P_u:
                    seed = None if base_seed is None else hash((base_seed, sim, p_h, p_e, p_u)) & 0xFFFFFFFF     #WTAF
                    jobs.append((p_h, p_e, p_u, sim, size_pop, T, seed, record_timeseries, fix_known_bugs))

    results = []
    t_start = time.perf_counter()
    report_every = max(1, len(jobs) // 20)
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(_sweep_worker, job) for job in jobs]
        completed = 0
        for fut in as_completed(futures):
            results.append(fut.result())
            completed += 1
            if progress and (completed % report_every == 0 or completed == len(jobs)):
                print(f"{completed}/{len(jobs)} runs complete ({time.perf_counter()-t_start:.1f}s elapsed)")

    summary_df = pd.DataFrame([{
        "P homeless": r.p_h, "P exit": r.p_e, "P undetected": r.p_u, "Simulation": r.sim,
        "# Consecutive": r.n_consecutive, "# Episodic": r.n_episodic, "# Chronic": r.n_chronic,
        "# Ever Homeless": r.n_ever_homeless, "# Undetected Chronic": r.n_undetected_chronic,
        "Frac Undetected Chronic": r.frac_undetected_chronic, "Wall Time (s)": r.wall_time_sec,
    } for r in results])
    summary_df["Episodic as % Chronic"] = summary_df["# Episodic"] / summary_df["# Chronic"]

    timeseries = {
        (r.sim, r.p_h, r.p_e, r.p_u): {
            "Times": r.times, "Now_homeless": r.now_homeless_ts, "Ever_homeless": r.ever_homeless_ts,
            "Consecutive_counts": r.consecutive_ts, "Episodic_counts": r.episodic_ts,
        } for r in results
    }
    print(f"Sweep complete: {len(jobs)} runs in {time.perf_counter()-t_start:.1f}s wall-clock "
          f"(sum of individual run times: {summary_df['Wall Time (s)'].sum():.1f}s)")
    return summary_df, timeseries


def save_sweep(summary_df, timeseries, out_dir="sweep_results", tag="sweep"):
    out_dir = Path(out_dir); out_dir.mkdir(exist_ok=True, parents=True)
    summary_path, ts_path = out_dir / f"{tag}_summary.csv", out_dir / f"{tag}_timeseries.pkl"
    summary_df.to_csv(summary_path, index=False)
    with open(ts_path, "wb") as f:
        pickle.dump(timeseries, f)
    print(f"Saved {summary_path} and {ts_path}")
    return summary_path, ts_path


def load_sweep(out_dir="sweep_results", tag="sweep"):
    out_dir = Path(out_dir)
    summary_df = pd.read_csv(out_dir / f"{tag}_summary.csv")
    with open(out_dir / f"{tag}_timeseries.pkl", "rb") as f:
        timeseries = pickle.load(f)
    return summary_df, timeseries
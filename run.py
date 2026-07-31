# run.py (parallelisiert mit --workers)
# -*- coding: utf-8 -*-
"""
Run LPP-CGN optimisation row-by-row from Data/config.csv and log results.

Neu:
- Parameter --workers N für parallele Ausführung (ThreadPool).
- Logger-Schreibzugriffe sind mit einem Lock geschützt.
"""

import os
import time
import traceback
import argparse
import pandas as pd
import shutil
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed

from load_data import load_and_build, load_candidate_config
from solve_cgn_one_stage import solve_one_stage
from solve_cgn_separated import solve_two_stage_separated
from solve_cgn_integrated import solve_two_stage_integrated
from solve_cgn_wait_and_see import solve_wait_and_see
from solve_cgn_eev import solve_eev, solve_two_stage_eev

from log import RunBatchLogger
from compute_bounds import compute_bounds


# Procedures that share a run group when expanded from a single config row.
_BOUNDS_PROCEDURES = ("ws", "integrated", "separated", "eev")


def _expand_procedures(proc_cell):
    """
    Parse the `procedure` cell. Returns a list of single procedures.

    Accepted forms:
      "integrated"                          → ["integrated"]
      "ws,integrated,separated,eev"         → list
      "bounds"                              → expansion to the four bounds-relevant procedures
    """
    s = str(proc_cell or "").strip().lower()
    if not s:
        return ["one"]
    if s == "bounds":
        return list(_BOUNDS_PROCEDURES)
    if "," in s:
        return [p.strip() for p in s.split(",") if p.strip()]
    return [s]


def _expand_cfg_df(cfg_df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand each input row into one row per procedure. All sub-rows of the
    same input row share a `run_group_id`, which `compute_bounds.py` uses
    to join them when producing `bounds_summary.csv`.
    """
    rows = []
    # Use enumerate (not the DataFrame index) so non-default indices
    # from upstream filters/concat still yield unique, contiguous group ids.
    for src_pos, (_, src) in enumerate(cfg_df.iterrows()):
        cfg_d = src.to_dict()
        procs = _expand_procedures(cfg_d.get("procedure", "one"))
        run_group_id = f"g_{src_pos:03d}"
        for proc in procs:
            row = dict(cfg_d)
            row["procedure"] = proc
            row["run_group_id"] = run_group_id
            rows.append(row)
    return pd.DataFrame(rows)

# ---------- helpers ----------

def _sum_over_scenarios(scen_list, key):
    return sum(float(s.get("prob") or 0.0) * float(s.get(key) or 0.0) for s in (scen_list or []))

def _agg_components_two_stage(solution):
    nom = solution.get("costs_0") or {}
    scen = solution.get("scenarios") or []

    agg_time      = float(nom.get("time")      or 0.0) + _sum_over_scenarios(scen, "cost_time")
    agg_time_base = float(nom.get("time_base") or 0.0) + _sum_over_scenarios(scen, "cost_time_base")
    agg_time_over = float(nom.get("time_over") or 0.0) + _sum_over_scenarios(scen, "cost_time_over")
    agg_bypass    = float(nom.get("bypass")    or 0.0) + _sum_over_scenarios(scen, "cost_bypass")
    agg_wait      = float(nom.get("wait")      or 0.0) + _sum_over_scenarios(scen, "cost_wait")
    agg_oper      = float(nom.get("oper")      or 0.0) + _sum_over_scenarios(scen, "cost_oper")
    return agg_time, agg_time_base, agg_time_over, agg_bypass, agg_wait, agg_oper

def _as_bool(x, default=False):
    if x is None:
        return default
    s = str(x).strip().lower()
    if s in ("1", "true", "yes", "y"): return True
    if s in ("0", "false", "no", "n"): return False
    return default

def _status_name(code):
    MAP = {2: "OPTIMAL", 3: "INFEASIBLE", 4: "INF_OR_UNBD", 5: "UNBOUNDED", 9: "TIME_LIMIT", 13: "SUBOPTIMAL"}
    try:
        if isinstance(code, str):
            return code
        c = int(code)
        return MAP.get(c, str(c))
    except Exception:
        return str(code)

def _normalize_artifacts(artifacts):
    arts = artifacts or {}
    cgn_stage1 = arts.get("cgn_stage1") or arts.get("cgn") or arts.get("cgn0")
    x_stage1   = arts.get("x_stage1")   or arts.get("x0")
    a2k_stage1 = arts.get("arc_to_keys_stage1") or arts.get("arc_to_keys0") or arts.get("arc_to_keys")

    cgn_s_list = (arts.get("cgn_stage2_list") or arts.get("cgn_s_list") or [])
    x_s_list   = (arts.get("x_stage2_list") or arts.get("xs") or arts.get("x_s_list") or [])
    a2k_s_list = (arts.get("arc_to_keys_stage2_list") or arts.get("arc_to_keys_s") or [])

    arts.setdefault("cgn_stage1", cgn_stage1)
    arts.setdefault("x_stage1", x_stage1)
    arts.setdefault("arc_to_keys_stage1", a2k_stage1)
    arts.setdefault("cgn_stage2_list", cgn_s_list)
    arts.setdefault("x_stage2_list", x_s_list)
    arts.setdefault("arc_to_keys_stage2_list", a2k_s_list)
    return arts

# ---------- Kernarbeit für eine Zeile (threadfähig) ----------

def run_one_row(i, cfg_row_dict, data_root, cand_cfg, logger, log_lock):
    """
    Führt einen kompletten Run für eine Config-Zeile aus.
    Rechnen: parallel
    Logger-Schreibzugriffe: mit log_lock geschützt.
    """
    tag = f"{cfg_row_dict.get('source')}/{cfg_row_dict.get('network')}@{cfg_row_dict.get('scenario_line_data')}"
    print(f"\n\n##### RUN {i}: {tag} #####")
    t0 = time.time()

    # Basiszeile für base_log (enthält alle Config-Spalten)
    base_row = logger.base_row_template(pd.Series(cfg_row_dict))

    try:
        # ----- 1) Load data & build model -----
        domain, model = load_and_build(
            data_root=data_root,
            cfg_row=cfg_row_dict,
            symmetrise_infra=False,
            zero_od_diagonal=False,
        )
        setattr(domain, "cand_cfg", cand_cfg)

        # Copy scenario files to run output folder
        run_folder = logger.run_dir(i)
        shutil.copy(domain.props["scenario_infra_file"], os.path.join(run_folder, "scenario_infra_used.csv"))
        shutil.copy(domain.props["scenario_prob_file"], os.path.join(run_folder, "scenario_prob_used.csv"))

        # ----- 2) Solve according to procedure -----
        proc = str(domain.config.get("procedure", "one")).lower()
        if proc in ("one", "one_stage"):
            m, solution, artifacts = solve_one_stage(domain, model)
        elif proc in ("integrated", "joint"):
            m, solution, artifacts = solve_two_stage_integrated(domain, model)
        elif proc in ("separated", "sequential"):
            m, solution, artifacts = solve_two_stage_separated(domain, model)
        elif proc in ("ws", "wait_and_see"):
            m, solution, artifacts = solve_wait_and_see(domain, model)
        elif proc in ("eev", "expected_value"):
            m, solution, artifacts = solve_eev(domain, model)
        elif proc in ("two_stage_eev", "two_stage_expected_value"):
            m, solution, artifacts = solve_two_stage_eev(domain, model)
        else:
            print(f"[WARN] unknown procedure '{proc}', falling back to one_stage")
            m, solution, artifacts = solve_one_stage(domain, model)

        # Normalize artifact keys so the rest of the script is solver-agnostic
        artifacts = _normalize_artifacts(artifacts)

        # ----- 3) KPIs & base_row -----
        # ----- 3) Aggregate KPIs & fill base row -----
        if proc in ("one", "one_stage"):
            nom = solution.get("costs_0") or {}
            base_row.update({
                "status": _status_name(solution.get("status")),
                "objective": solution.get("objective"),
                "runtime_s": solution.get("runtime_s"),
                "opt_gap": solution.get("opt_gap"),
                "cost_time": nom.get("time"),
                "cost_time_base": nom.get("time_base"),
                "cost_time_over": nom.get("time_over"),
                "cost_bypass": nom.get("bypass"),
                "cost_wait": nom.get("wait"),
                "cost_oper": nom.get("oper"),
                "obj_stage1": nom.get("objective"),
                "obj_stage2_exp": None,
                "repl_cost_freq_exp": None,
                "repl_cost_path_exp": None,
                "repl_cost_exp": None,
            })

            # Frequencies + paths header even for one-stage (scenarios=[])
            logger.write_freqs_two_stage(
                i, model,
                nominal=(solution.get("chosen_freq")
                         or solution.get("chosen_freq_stage1")
                         or artifacts.get("chosen_freq")
                         or artifacts.get("chosen_freq_stage1")
                         or {}),
                scenarios=[],
                nominal_costs=solution.get("costs_0"),
            )

        else:
            # Properly aggregate (includes base/over split and bypass)
            agg_time, agg_time_base, agg_time_over, agg_bypass, agg_wait, agg_oper = _agg_components_two_stage(solution)

            # If a two-stage solver returned no objective at all (e.g. stage-1
            # infeasible early-return from separated or EEV), zero costs from
            # the aggregator are misleading — null them so base_log stays
            # consistent with the missing objective.
            if solution.get("objective") is None:
                agg_time = agg_time_base = agg_time_over = None
                agg_bypass = agg_wait = agg_oper = None

            base_row.update({
                "status": _status_name(solution.get("status")),
                "objective": solution.get("objective"),
                "runtime_s": solution.get("runtime_s"),
                "opt_gap": solution.get("opt_gap"),
                "cost_time": agg_time,
                "cost_time_base": agg_time_base,
                "cost_time_over": agg_time_over,
                "cost_bypass": agg_bypass,
                "cost_wait": agg_wait,
                "cost_oper": agg_oper,
                "obj_stage1": solution.get("obj_stage1"),
                "obj_stage2_exp": solution.get("obj_stage2_exp"),
                "repl_cost_freq_exp": solution.get("repl_cost_freq_exp"),
                "repl_cost_path_exp": solution.get("repl_cost_path_exp"),
                "repl_cost_exp": solution.get("repl_cost_exp"),
            })

            # Wide frequency CSV incl. nominal costs and scenario paths
            logger.write_freqs_two_stage(
                i, model,
                nominal=solution.get("chosen_freq_stage1", {}) or {},
                scenarios=solution.get("scenarios", []) or [],
                nominal_costs=solution.get("costs_0"),
                cand_selected=(artifacts.get("cand_selected_lines")
                               or artifacts.get("cand_selected")),
                cand_all=(artifacts.get("candidates_lines")
                          or artifacts.get("candidates")),
            )
            # WS has no candidate-selection step; skip writing an empty CSV.
            if proc not in ("ws", "wait_and_see"):
                logger.write_candidates(
                    i,
                    model,
                    candidates_per_s=(artifacts.get("candidates_lines")
                                      or artifacts.get("candidates")
                                      or {}),
                    c_repl_line=float(domain.config.get("cost_repl_line", 0.0)),
                    selected=(artifacts.get("cand_selected_lines")
                              or artifacts.get("cand_selected")
                              or {}),
                    freqs_per_s=(solution.get("chosen_freq_stage2") or []),
                )

        # ----- 4) Edge flows logs -----
        with log_lock:
            if artifacts.get("cgn_stage1") is not None and artifacts.get("x_stage1") is not None:
                logger.write_edge_passenger_flows(
                    i, model,
                    artifacts["cgn_stage1"], artifacts["x_stage1"],
                    arc_to_keys=artifacts.get("arc_to_keys_stage1"),
                    filename_suffix="_stage1",
                    links_df=domain.links_df
                )

            cgn_s_list = artifacts.get("cgn_stage2_list") or []
            x_s_list   = artifacts.get("x_stage2_list") or []
            a2k_s_list = artifacts.get("arc_to_keys_stage2_list") or []
            for s, (cgn_s, x_s) in enumerate(zip(cgn_s_list, x_s_list)):
                a2k_s = a2k_s_list[s] if s < len(a2k_s_list) else None
                logger.write_edge_passenger_flows(
                    i, model, cgn_s, x_s,
                    arc_to_keys=a2k_s,
                    filename_suffix=f"_stage2_{s}",
                    links_df=domain.links_df
                )

        # Console summary
        print(
            f"Status={_status_name(solution.get('status'))}  "
            f"Obj={solution.get('objective')}  "
            f"Stage1={base_row.get('obj_stage1')}  "
            f"Stage2_exp={base_row.get('obj_stage2_exp')}  "
            f"Repl_exp={base_row.get('repl_cost_exp')}  "
            f"Runtime={solution.get('runtime_s')}s"
        )

        status_code = int(solution.get("status_code", -1))
        base_row["status_code"] = status_code

    except Exception as e:
        traceback.print_exc()
        base_row.update({
            "status_code": -1,
            "status": "ERROR",
            "objective": None,
            "runtime_s": round(time.time() - t0, 3),
            "cost_time": None,
            "cost_wait": None,
            "cost_oper": None,
        })
        print(f"[ERROR] run {i} ({tag}): {e}")

    # Base row IMMER am Ende anhängen (geschützt)
    with log_lock:
        logger.append_base_row(base_row)

# ---------- main ----------

def parse_args():
    ap = argparse.ArgumentParser(description="Run LPP-CGN optimisation over config.csv")
    ap.add_argument("--workers", type=int, default=1, help="Anzahl paralleler Läufe (Threads). Standard: 1")
    ap.add_argument("--data-root", type=str, default=".", help="Wurzelverzeichnis (enthält Data/ und Results/)")
    ap.add_argument("--config", type=str, default=None,
                    help="Alternative Config-CSV (Default: Data/config.csv). "
                         "Akzeptiert absolute oder relative Pfade.")
    ap.add_argument("--row", type=int, default=None,
                    help="Nur diese Zeile (0-basiert) aus der INPUT-Config laufen. "
                         "Procedure-Expansion findet weiterhin statt. Nützlich für SLURM-Arrays.")
    ap.add_argument("--stamp-suffix", type=str, default=None,
                    help="Suffix für den Lauf-Ordnernamen, z.B. ${SLURM_ARRAY_JOB_ID}_${TASK} "
                         "damit parallele Array-Tasks eindeutige Ordner bekommen. "
                         "WICHTIG: ohne Suffix können parallele run.py-Aufrufe in derselben "
                         "Minute auf denselben Ordner schreiben und sich gegenseitig "
                         "base_log.csv-Zeilen überschreiben.")
    ap.add_argument("--results-subdir", type=str, default=None,
                    help="Optionaler Subfolder unter Results/, z.B. 'job_<JOBID>' für "
                         "SLURM-Array-Jobs. Output landet dann in "
                         "Results/<subdir>/<stamp>/ statt direkt unter Results/.")
    return ap.parse_args()

def main():
    args = parse_args()
    data_root = args.data_root
    cfg_path = args.config if args.config else os.path.join(data_root, "Data", "config.csv")
    cfg_df    = pd.read_csv(cfg_path, sep=';')
    # Visible audit: which file did we actually read? Catches the failure
    # mode where the sbatch picked up a stale expanded config and silently
    # ran the wrong parameters.
    print(f"[run.py] config={cfg_path}, rows={len(cfg_df)}")
    for k in ("num_od", "waiting_time_frequency", "bypass_multiplier", "demand_scale"):
        if k in cfg_df.columns:
            uniq = cfg_df[k].unique()
            print(f"[run.py]   {k} = {list(uniq)}")

    # Optional row-selection BEFORE procedure expansion: an array task that
    # passes --row N gets exactly the procedures of input row N (which may
    # itself expand to multiple sub-runs via procedure=bounds).
    if args.row is not None:
        if args.row < 0 or args.row >= len(cfg_df):
            raise IndexError(
                f"--row {args.row} out of range for config with {len(cfg_df)} rows"
            )
        cfg_df = cfg_df.iloc[[args.row]].reset_index(drop=True)

    # Expand procedure lists/aliases into one row per actual solver run.
    # Adds `run_group_id` so bounds aggregation can join sibling runs later.
    cfg_df = _expand_cfg_df(cfg_df).reset_index(drop=True)

    # Kandidaten-Konfig einmal laden (read-only, für alle Threads verwendbar)
    cand_cfg = load_candidate_config(data_root)

    # Run-folder stamp: default = current time. With --stamp-suffix it becomes
    # `<datetime>_<suffix>` so SLURM array tasks land in distinct folders.
    stamp = None
    if args.stamp_suffix:
        from datetime import datetime
        stamp = datetime.now().strftime("%Y_%m_%d_%H_%M") + "_" + str(args.stamp_suffix)

    # Zentraler Logger (legt einmal den Lauf-Ordner an)
    logger = RunBatchLogger(data_root=data_root, cfg_df=cfg_df,
                            stamp=stamp, subdir=args.results_subdir)
    print(f"Logging to: {logger.out_dir}")

    # Lock für alle Logger-Schreibzugriffe
    log_lock = Lock()

    # Sequentiell
    if args.workers <= 1 or len(cfg_df) <= 1:
        for i, cfg_row in cfg_df.iterrows():
            run_one_row(i, cfg_row.to_dict(), data_root, cand_cfg, logger, log_lock)

    # Parallel (ThreadPool)
    else:
        max_workers = max(1, int(args.workers))
        print(f"Starte parallel mit {max_workers} Worker-Threads…")
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = []
            for i, cfg_row in cfg_df.iterrows():
                futures.append(
                    ex.submit(
                        run_one_row,
                        i, cfg_row.to_dict(), data_root, cand_cfg, logger, log_lock
                    )
                )
            # Optional: auf Abschluss warten & Fehler sofort zeigen
            for f in as_completed(futures):
                try:
                    f.result()
                except Exception:
                    traceback.print_exc()

    print(f"\nBase log: {logger.base_log_path}")

    # End-of-batch hook: aggregate WS/RP/EEV_nom/EEV_mean per run_group_id and
    # write `bounds_summary.csv` next to base_log.csv. Incomplete groups still
    # produce a row with NaN in columns whose ingredients are missing.
    try:
        bounds_path = compute_bounds(logger.out_dir)
        if bounds_path:
            print(f"Bounds summary: {bounds_path}")
    except Exception:
        traceback.print_exc()

if __name__ == "__main__":
    main()

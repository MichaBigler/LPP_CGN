# run.py
# -*- coding: utf-8 -*-
"""
Run LPP-CGN optimisation row-by-row from Data/config.csv and log results.

Pipeline per row:
1) Load data & build model
2) Solve (one-stage / integrated two-stage / separated two-stage)
3) Aggregate KPIs
4) Write logs:
   - Results/<stamp>/base_log.csv (one row per config line)
   - Results/<stamp>/row_XXX/freq.csv
   - Results/<stamp>/row_XXX/candidates.csv (for 2-stage only)
   - Results/<stamp>/row_XXX/edge_flows_stage1.csv
   - Results/<stamp>/row_XXX/edge_flows_stage2_s.csv  (one per scenario)
"""

import os
import time
import traceback
import pandas as pd

from load_data import load_and_build, load_candidate_config
from solve_cgn_one_stage import solve_one_stage
from solve_cgn_separated import solve_two_stage_separated
from solve_cgn_integrated import solve_two_stage_integrated

from log import RunBatchLogger


# ---------- helpers ----------

def _sum_over_scenarios(scen_list, key):
    """Probability-weighted sum over scenarios for a given key."""
    return sum(float(s.get("prob") or 0.0) * float(s.get(key) or 0.0) for s in (scen_list or []))


def _agg_components_two_stage(solution):
    """
    Aggregate (prob.-weighted) Stage-1 + Stage-2 component costs.
    Returns: (time, time_base, time_over, bypass, wait, oper).
    Assumes scenario dicts store ALREADY WEIGHTED components (time_w, wait_w, op_w applied).
    """
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
    """Map common Gurobi status codes to readable names."""
    MAP = {
        2: "OPTIMAL",
        3: "INFEASIBLE",
        4: "INF_OR_UNBD",
        5: "UNBOUNDED",
        9: "TIME_LIMIT",
        13: "SUBOPTIMAL",
    }
    try:
        if isinstance(code, str):
            return code
        c = int(code)
        return MAP.get(c, str(c))
    except Exception:
        return str(code)


def _normalize_artifacts(artifacts):
    """
    Make artifact keys consistent for the rest of this script.
    Some solvers return (cgn/x) as *_stage1, others as cgn/x or cgn0/x0; same for stage2 lists.
    This function maps all those variants to a single canonical set:
      - cgn_stage1, x_stage1, arc_to_keys_stage1
      - cgn_stage2_list, x_stage2_list, arc_to_keys_stage2_list
    """
    arts = artifacts or {}

    # Stage-1
    cgn_stage1 = arts.get("cgn_stage1") or arts.get("cgn") or arts.get("cgn0")
    x_stage1   = arts.get("x_stage1")   or arts.get("x0")
    a2k_stage1 = arts.get("arc_to_keys_stage1") or arts.get("arc_to_keys0") or arts.get("arc_to_keys")

    # Stage-2 lists
    cgn_s_list = (arts.get("cgn_stage2_list")
                  or arts.get("cgn_s_list")
                  or [])
    x_s_list   = (arts.get("x_stage2_list")
                  or arts.get("xs") or arts.get("x_s_list")
                  or [])
    a2k_s_list = (arts.get("arc_to_keys_stage2_list")
                  or arts.get("arc_to_keys_s")
                  or [])

    # Write back normalized keys (non-destructive)
    arts.setdefault("cgn_stage1", cgn_stage1)
    arts.setdefault("x_stage1", x_stage1)
    arts.setdefault("arc_to_keys_stage1", a2k_stage1)
    arts.setdefault("cgn_stage2_list", cgn_s_list)
    arts.setdefault("x_stage2_list", x_s_list)
    arts.setdefault("arc_to_keys_stage2_list", a2k_s_list)

    return arts


# ---------- main ----------

def main():
    data_root = os.environ.get("DATA_ROOT", ".")
    cfg_path  = os.path.join(data_root, "Data", "config.csv")
    cfg_df    = pd.read_csv(cfg_path, sep=';')

    cand_cfg = load_candidate_config(data_root)

    # Central logger: creates Results/<YYYY_MM_DD_HH_MM>/ and base_log.csv
    logger = RunBatchLogger(data_root=data_root, cfg_df=cfg_df)
    print(f"Logging to: {logger.out_dir}")

    for i, cfg_row in cfg_df.iterrows():
        tag = f"{cfg_row['source']}/{cfg_row['network']}@{cfg_row['scenario_line_data']}"
        print(f"\n\n##### RUN {i}: {tag} #####")
        t0 = time.time()

        # Prepare base row for base_log.csv (contains all config columns)
        base_row = logger.base_row_template(cfg_row)

        try:
            # ----- 1) Load data & build model -----
            domain, model = load_and_build(
                data_root=data_root,
                cfg_row=cfg_row.to_dict(),
                symmetrise_infra=False,
                zero_od_diagonal=False,
            )
            setattr(domain, "cand_cfg", cand_cfg)

            # ----- 2) Solve according to procedure -----
            proc = str(domain.config.get("procedure", "one")).lower()
            if proc in ("one", "one_stage"):
                m, solution, artifacts = solve_one_stage(domain, model)
            elif proc in ("integrated", "joint"):
                m, solution, artifacts = solve_two_stage_integrated(domain, model)
            elif proc in ("separated", "sequential"):
                m, solution, artifacts = solve_two_stage_separated(domain, model)
            else:
                print(f"[WARN] unknown procedure '{proc}', falling back to one_stage")
                m, solution, artifacts = solve_one_stage(domain, model)

            # Normalize artifact keys so the rest of the script is solver-agnostic
            artifacts = _normalize_artifacts(artifacts)

            # ----- 3) Aggregate KPIs & fill base row -----
            if proc in ("one", "one_stage"):
                nom = solution.get("costs_0") or {}
                base_row.update({
                    "status": _status_name(solution.get("status")),
                    "objective": solution.get("objective"),
                    "runtime_s": solution.get("runtime_s"),
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

                base_row.update({
                    "status": _status_name(solution.get("status")),
                    "objective":      solution.get("objective"),
                    "runtime_s":      solution.get("runtime_s"),
                    "cost_time":      agg_time,
                    "cost_time_base": agg_time_base,
                    "cost_time_over": agg_time_over,
                    "cost_bypass":    agg_bypass,
                    "cost_wait":      agg_wait,
                    "cost_oper":      agg_oper,
                    "obj_stage1":     solution.get("obj_stage1"),
                    "obj_stage2_exp": solution.get("obj_stage2_exp"),
                    "repl_cost_freq_exp": solution.get("repl_cost_freq_exp"),
                    "repl_cost_path_exp": solution.get("repl_cost_path_exp"),
                    "repl_cost_exp":      solution.get("repl_cost_exp"),
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

                # Candidates CSV (2-stage only)
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

            # ----- 4) Edge flows logs (stage 1 + stage 2 per scenario) -----
            # Stage 1
            if artifacts.get("cgn_stage1") is not None and artifacts.get("x_stage1") is not None:
                logger.write_edge_passenger_flows(
                    i, model,
                    artifacts["cgn_stage1"], artifacts["x_stage1"],
                    arc_to_keys=artifacts.get("arc_to_keys_stage1"),
                    filename_suffix="_stage1",
                )

            # Stage 2 per scenario
            cgn_s_list = artifacts.get("cgn_stage2_list") or []
            x_s_list   = artifacts.get("x_stage2_list") or []
            a2k_s_list = artifacts.get("arc_to_keys_stage2_list") or []
            for s, (cgn_s, x_s) in enumerate(zip(cgn_s_list, x_s_list)):
                a2k_s = a2k_s_list[s] if s < len(a2k_s_list) else None
                logger.write_edge_passenger_flows(
                    i, model, cgn_s, x_s,
                    arc_to_keys=a2k_s,
                    filename_suffix=f"_stage2_{s}",
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

        except Exception as e:
            # Ensure a meaningful base_log row even on failure
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

        finally:
            # Always append base row (streaming log)
            if 'solution' in locals() and isinstance(solution, dict):
                base_row["status_code"] = int(solution.get("status_code", -1))
            else:
                base_row["status_code"] = -1
            logger.append_base_row(base_row)

    print(f"\nBase log: {logger.base_log_path}")


if __name__ == "__main__":
    main()

#run.py
# -*- coding: utf-8 -*-
"""Run LPP-CGN optimisation row-by-row from Data/config.csv, logging via log.py."""

import os
import time
import traceback
import pandas as pd

from load_data import load_and_build
from load_data import load_candidate_config
from solve_cgn_one_stage import solve_one_stage
from solve_cgn_separated import solve_two_stage_separated
from solve_cgn_integrated import solve_two_stage_integrated
from solve_utils import _freq_values_from_config  

from print import print_domain_summary, print_model_summary  # optional
from log import RunBatchLogger

def _agg_components_two_stage(solution):
    nom = solution.get("costs_0") or {}
    scen_list = solution.get("scenarios") or []

    def _sum_scens(key):
        return sum(float(s.get("prob") or 0.0) * float(s.get(key) or 0.0) for s in scen_list)

    agg_time      = float(nom.get("time")      or 0.0) + _sum_scens("cost_time")
    agg_time_base = float(nom.get("time_base") or 0.0) + _sum_scens("cost_time_base")
    agg_time_over = float(nom.get("time_over") or 0.0) + _sum_scens("cost_time_over")
    agg_bypass = float(nom.get("bypass") or 0.0) + _sum_scens("cost_bypass")
    agg_wait = float(nom.get("wait") or 0.0) + _sum_scens("cost_wait")
    agg_oper = float(nom.get("oper") or 0.0) + _sum_scens("cost_oper")
    return agg_time, agg_time_base, agg_time_over, agg_bypass, agg_wait, agg_oper


def _as_bool(x, default=False):
    if x is None:
        return default
    s = str(x).strip().lower()
    if s in ("1", "true", "yes", "y"): return True
    if s in ("0", "false", "no", "n"): return False
    return default

def _status_name(code):
    # map the common Gurobi status codes to names
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

def main():
    data_root = os.environ.get("DATA_ROOT", ".")
    cfg_path  = os.path.join(data_root, "Data", "config.csv")
    cfg_df    = pd.read_csv(cfg_path, sep=';')

    cand_cfg = load_candidate_config(data_root)


    # optional hübsche Konsolenprints
    verbose = _as_bool(os.environ.get("VERBOSE_PRINTS", "0"), default=False)

    # zentraler Logger: legt Results/<YYYY_MM_DD_HH_MM>/ an und schreibt base_log.csv (Header)
    logger = RunBatchLogger(data_root=data_root, cfg_df=cfg_df)
    print(f"Logging to: {logger.out_dir}")

    for i, cfg_row in cfg_df.iterrows():
        tag = f"{cfg_row['source']}/{cfg_row['network']}@{cfg_row['scenario_line_data']}"
        print(f"\n\n##### RUN {i}: {tag} #####")
        t0 = time.time()

        # Basizeile fürs Base-Log: enthält bereits alle config-Spalten
        base_row = logger.base_row_template(cfg_row)

        try:
            domain, model = load_and_build(
                data_root=data_root,
                cfg_row=cfg_row.to_dict(),
                symmetrise_infra=False,
                zero_od_diagonal=False,
            )
            setattr(domain, "cand_cfg", cand_cfg)

            if verbose:
                print_domain_summary(domain, max_rows=5, max_lines=5, max_include=5)
                print_model_summary(model, domain, max_items=5, max_lines=5)

            proc = str(domain.config.get("procedure", "one")).lower()
            if proc in ("one", "one_stage"):
                m, solution, artifacts = solve_one_stage(domain, model)
            else:
                if proc in ("integrated", "joint"):
                    m, solution, artifacts = solve_two_stage_integrated(domain, model)
                elif proc in ("separated", "sequential"):
                    m, solution, artifacts = solve_two_stage_separated(domain, model)
                else:
                    print(f"[WARN] unknown procedure '{proc}', falling back to one_stage")
                    m, solution, artifacts = solve_one_stage(domain, model)


            if proc in ("one", "one_stage"):
                nom = (solution.get("costs_0") or {})
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

                logger.write_freqs_two_stage(
                    i, model,
                    nominal=(
                        solution.get("chosen_freq")
                        or solution.get("chosen_freq_stage1")
                        or artifacts.get("chosen_freq")
                        or artifacts.get("chosen_freq_stage1")
                        or {}
                    ),
                    scenarios=[],
                    nominal_costs=solution.get("costs_0"),
                )
                logger.write_edge_passenger_flows(
                    i, model,
                    artifacts["cgn_stage1"], artifacts["x_stage1"],
                    arc_to_keys=artifacts.get("arc_to_keys_stage1"),
                    filename_suffix="_stage1",
                )
            else:
                # sauber aggregieren (inkl. Base/Over-Split & Bypass)
                agg_time, agg_time_base, agg_time_over, agg_bypass, agg_wait, agg_oper = _agg_components_two_stage(solution)
                
                base_row.update({
                    "status": _status_name(solution.get("status")),
                    "objective":     solution.get("objective"),
                    "runtime_s":     solution.get("runtime_s"),
                    "cost_time":     agg_time,
                    "cost_time_base": agg_time_base,
                    "cost_time_over": agg_time_over,
                    "cost_bypass":   agg_bypass,
                    "cost_wait":     agg_wait,
                    "cost_oper":     agg_oper,
                    "obj_stage1":    solution.get("obj_stage1"),
                    "obj_stage2_exp": solution.get("obj_stage2_exp"),
                    "repl_cost_freq_exp": solution.get("repl_cost_freq_exp"),
                    "repl_cost_path_exp": solution.get("repl_cost_path_exp"),
                    "repl_cost_exp":  solution.get("repl_cost_exp"),
                })
                # breite Frequenz-CSV inkl. Nominal-Kosten
                logger.write_freqs_two_stage(
                    i, model,
                    nominal=solution.get("chosen_freq_stage1", {}) or {},
                    scenarios=solution.get("scenarios", []) or [],
                    nominal_costs=solution.get("costs_0"),
                    cand_selected=artifacts.get("cand_selected") or artifacts.get("cand_selected_lines"),
                    cand_all=artifacts.get("candidates") or artifacts.get("candidates_lines"),
                )
                cand_per_s  = artifacts.get("candidates") or artifacts.get("candidates_lines") or {}
                sel_per_s   = artifacts.get("cand_selected") or artifacts.get("cand_selected_lines") or {}
                freqs_per_s = solution.get("chosen_freq_stage2") or []

                logger.write_candidates(
                    i,
                    model,
                    candidates_per_s = artifacts.get("candidates") or artifacts.get("candidates_lines") or {},
                    c_repl_line      = float(domain.config.get("cost_repl_line", 0.0)),
                    selected         = artifacts.get("cand_selected") or artifacts.get("cand_selected_lines") or {},
                    freqs_per_s      = solution.get("chosen_freq_stage2") or [],
                )


                # Stage 1
                logger.write_edge_passenger_flows(
                    i, model,
                    artifacts["cgn_stage1"], artifacts["x_stage1"],
                    arc_to_keys=artifacts.get("arc_to_keys_stage1"),
                    filename_suffix="_stage1",
                )
                
                # Stage 2 je Szenario
                for s, (cgn_s, x_s) in enumerate(zip(artifacts.get("cgn_stage2_list", []), artifacts.get("x_stage2_list", []))):
                    a2k_s = None
                    a2k_list = artifacts.get("arc_to_keys_stage2_list", [])
                    if s < len(a2k_list): a2k_s = a2k_list[s]
                    print(f"[flows-debug] writing stage2 flows for run {i} scenario {s} with {len(x_s)} x-entries and arc_to_keys={a2k_s is not None}")
                    print(a2k_list)
                    logger.write_edge_passenger_flows(
                        i, model, cgn_s, x_s,
                        arc_to_keys=a2k_s,
                        filename_suffix=f"_stage2_{s}",
                    )

            print(f"Status={_status_name(solution.get('status'))}  Obj={solution.get('objective')}  "
                f"Stage1={base_row.get('obj_stage1')}  Stage2_exp={base_row.get('obj_stage2_exp')}  "
                f"Repl_exp={base_row.get('repl_cost_exp')}  Runtime={solution.get('runtime_s')}s")

        except Exception as e:
            # Im Fehlerfall trotzdem eine sinnvolle Zeile loggen
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
            # Sofort anhängen (Streaming-Log)
            if 'solution' in locals() and isinstance(solution, dict):
                base_row["status_code"] = int(solution.get("status_code", -1))
            else:
                base_row["status_code"] = -1

            try:
                if artifacts and isinstance(artifacts, dict):
                    # Stage 1
                    cgn0 = artifacts.get("cgn_stage1") or artifacts.get("cgn")
                    x0   = artifacts.get("x_stage1")   or artifacts.get("x0")
                    atk0 = (artifacts.get("arc_to_keys_stage1")
                            or artifacts.get("arc_to_keys0")
                            or artifacts.get("arc_to_keys"))
                    if cgn0 is not None and x0 is not None:
                        logger.write_edge_passenger_flows(
                            i, model, cgn0, x0,
                            arc_to_keys=atk0, filename_suffix="_stage1"
                        )

                    # Stage 2 je Szenario
                    cgn_s_list = artifacts.get("cgn_stage2_list") or []
                    xs_list    = artifacts.get("x_stage2_list") or []
                    atk_s_list = artifacts.get("arc_to_keys_stage2_list") or []
                    # Debug: zeig Längen statt ganzer Objekte
                    print(f"[flows-debug] stage2 counts: cgn={len(cgn_s_list)} x={len(xs_list)} atk={len(atk_s_list)}")

                    for s, (cgn_s, x_s) in enumerate(zip(cgn_s_list, xs_list)):
                        atk_s = atk_s_list[s] if s < len(atk_s_list) else None
                        logger.write_edge_passenger_flows(
                            i, model, cgn_s, x_s,
                            arc_to_keys=atk_s, filename_suffix=f"_stage2_{s}"
                        )
            except Exception as _e:
                print(f"[WARN] edge-flow logging failed for run {i}: {_e}")

            logger.append_base_row(base_row)

    print(f"\nBase log: {logger.base_log_path}")


if __name__ == "__main__":
    main()

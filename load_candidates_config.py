# load_candidates_cfg.py
import os
import pandas as pd
from data_model import CandidateConfig

_BOOL_TRUE = {"1","true","yes","y","on","t"}
_BOOL_FALSE = {"0","false","no","n","off","f"}

def _as_bool(x, default=False):
    if x is None: return default
    s = str(x).strip().lower()
    if s in _BOOL_TRUE: return True
    if s in _BOOL_FALSE: return False
    return default

def _as_opt_float(x):
    if x is None or str(x).strip()=="":
        return None
    return float(x)

def _norm_mirror(x: str) -> str:
    s = (x or "").strip().lower()
    return s if s in ("auto","force","off") else "auto"

def load_candidate_config(data_root: str) -> CandidateConfig:
    path = os.path.join(data_root, "Data", "config_candidates.csv")
    if not os.path.exists(path):
        # Datei optional – wenn nicht vorhanden, Defaults
        return CandidateConfig()

    df = pd.read_csv(path, sep=';')
    if df.empty:
        return CandidateConfig()

    r = df.iloc[0]  # global: erste Zeile gilt für alle Runs

    return CandidateConfig(
        k_loc_detour           = int(r.get("k_loc_detour", 3)),
        k_sp_global            = int(r.get("k_sp_global", 8)),
        max_candidates_per_line= int(r.get("max_candidates_per_line", 20)),
        div_min_edges          = int(r.get("div_min_edges", 1)),
        w_len                  = _as_opt_float(r.get("w_len")),
        w_repl                 = _as_opt_float(r.get("w_repl")),
        corr_eps               = float(r.get("corr_eps", 0.25)),
        generate_only_if_disrupted = _as_bool(r.get("generate_only_if_disrupted"), True),
        mirror_backward        = _norm_mirror(r.get("mirror_backward", "auto")),
    )

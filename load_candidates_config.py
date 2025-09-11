# load_candidate_config.py
import os, pandas as pd
from data_model import CandidateConfig

_BOOL_TRUE = {"1","true","y","yes"}
def _as_bool(x, default=False):
    if x is None: return default
    s = str(x).strip().lower()
    return s in _BOOL_TRUE if s in _BOOL_TRUE|{"0","false","n","no"} else default

def load_candidate_config(data_root: str) -> CandidateConfig:
    path = os.path.join(data_root, "Data", "config_candidates.csv")
    if not os.path.exists(path):
        return CandidateConfig()  # Defaults

    df = pd.read_csv(path, sep=';')
    if df.empty:
        return CandidateConfig()

    row = df.iloc[0].to_dict()
    return CandidateConfig(
        k_loc_detour          = int(row.get("k_loc_detour", 3)),
        k_sp_global           = int(row.get("k_sp_global", 8)),
        max_candidates_per_line = int(row.get("max_candidates_per_line", 20)),
        div_min_edges         = int(row.get("div_min_edges", 1)),
        w_len                 = (None if pd.isna(row.get("w_len", None)) else float(row.get("w_len"))),
        w_repl                = (None if pd.isna(row.get("w_repl", None)) else float(row.get("w_repl"))),
        corr_eps              = float(row.get("corr_eps", 0.25)),
        generate_only_if_disrupted = _as_bool(row.get("generate_only_if_disrupted", True), True),
        mirror_backward       = str(row.get("mirror_backward", "auto")).strip().lower(),
    )

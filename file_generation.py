import os
import pandas as pd

# What to generate
GENERATE_SCENARIO_INFRA = True
GENERATE_CONFIG = True
EXCLUDE_EDGES = [30, 16, 23, 17]

# ============================================================================
# Configuration
# ============================================================================
DATA_ROOT = os.environ.get("DATA_ROOT", ".")
NETWORK_DIR = os.path.join(DATA_ROOT, "lintim", "SiouxFalls")
SCEN_DIR = os.path.join(DATA_ROOT, "Data", "SiouxFalls")
GENERATED_DIR = os.path.join(SCEN_DIR, "generated")
CONFIG_OUTPUT = os.path.join(DATA_ROOT, "Data", "config.csv")

# Capacity multipliers per scenario
CAPACITY_LEVELS = [("*0.8", 1), ("*0.6", 2), ("*0.4", 3), ("*0.2", 4), ("0", 5)]

# Base config values (same for all runs)
BASE_CONFIG = {
    "source": "lintim",
    "network": "SiouxFalls",
    "scenario_line_data": "SiouxFalls",
    "routing_agg": "TRUE",
    "waiting_time_frequency": "False",
    "gap": 0.00001,
    "time_limit": 6000,
    "travel_time_cost_mult": 1,
    "waiting_time_cost_mult": 2,
    "line_operation_cost_mult": 20,
    "cost_repl_freq": 50,
    "cost_repl_line": 200,
    "repl_budget": -5,
    "bypass_multiplier": -1,
    "overdemand_threshold": 0.75,
    "overdemand_multiplier": 2,
    "num_od": 55,
    "train_capacity": 50,
    "infrastructure_capacity": 10,
    "max_frequency": 10,
    "scenario_infra_id": None,   # default: use scenario_infra.csv
    "scenario_prob_id": None,    # default: use scenario_prob.csv
}

# ============================================================================
# Step 1: Generate scenario_infra files
# ============================================================================
def generate_scenario_infra_files(edge_giv_path: str, generated_dir: str):
    """Generate one scenario_infra_{edge_id}.csv per edge in Edge.giv."""
    os.makedirs(generated_dir, exist_ok=True)

    edges_df = pd.read_csv(edge_giv_path, sep=';', comment='#', header=None,
                           names=['id', 'a', 'b', 'length', 't_min', 't_max'])
    edges_df = edges_df.astype({'id': int, 'a': int, 'b': int})

    # Separate excluded edges
    excluded_edges = edges_df[edges_df['id'].isin(EXCLUDE_EDGES)]
    normal_edges = edges_df[~edges_df['id'].isin(EXCLUDE_EDGES)]

    for _, r in normal_edges.iterrows():
        eid = int(r['id'])
        a, b = int(r['a']), int(r['b'])

        rows = []
        for cap, scen_id in CAPACITY_LEVELS:
            # Add the main edge
            rows.append({"scenario": scen_id, "left-stop": a, "right-stop": b, "infrastructure_capacity": cap})
            rows.append({"scenario": scen_id, "left-stop": b, "right-stop": a, "infrastructure_capacity": cap})
            # Add all excluded edges at the same capacity level
            for _, excl_r in excluded_edges.iterrows():
                ea, eb = int(excl_r['a']), int(excl_r['b'])
                rows.append({"scenario": scen_id, "left-stop": ea, "right-stop": eb, "infrastructure_capacity": cap})
                rows.append({"scenario": scen_id, "left-stop": eb, "right-stop": ea, "infrastructure_capacity": cap})

        df = pd.DataFrame(rows, columns=["scenario", "left-stop", "right-stop", "infrastructure_capacity"])
        out_path = os.path.join(generated_dir, f"scenario_infra_{eid}.csv")
        df.to_csv(out_path, sep=';', index=False)
        print(f"Generated: {out_path}")

    print(f"\nGenerated {len(edges_df)} scenario_infra files in {generated_dir}")
    return edges_df

# ============================================================================
# Step 2: Generate config.csv
# ============================================================================
def generate_config(edge_giv_path: str, config_output: str):
    """Generate config.csv with two rows per edge (integrated + separated)."""
    edges_df = pd.read_csv(edge_giv_path, sep=';', comment='#', header=None,
                           names=['id', 'a', 'b', 'length', 't_min', 't_max'])
    edges_df = edges_df.astype({'id': int, 'a': int, 'b': int})

    # Skip excluded edges
    normal_edges = edges_df[~edges_df['id'].isin(EXCLUDE_EDGES)]

    rows = []
    for _, r in normal_edges.iterrows():
        eid = int(r['id'])
        for procedure in ("separated",): #("integrated", "separated")
            for waiting in {False}:
                row = BASE_CONFIG.copy()
                row["procedure"] = procedure
                row["scenario_infra_id"] = eid
                row["scenario_prob_id"] = 1
                row["bypass_multiplier"] = 10
                row["waiting_time_frequency"] = waiting
                rows.append(row)

    # Column order must match config.csv header
    columns = [
        "source", "network", "scenario_line_data", "procedure",
        "routing_agg", "waiting_time_frequency", "gap", "time_limit",
        "travel_time_cost_mult", "waiting_time_cost_mult", "line_operation_cost_mult",
        "cost_repl_freq", "cost_repl_line", "repl_budget", "bypass_multiplier",
        "overdemand_threshold", "overdemand_multiplier", "num_od", "train_capacity",
        "infrastructure_capacity", "max_frequency", "scenario_infra_id", "scenario_prob_id"
    ]

    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(config_output, sep=';', index=False)
    print(f"Generated config.csv with {len(df)} rows at {config_output}")

# Main - configure the files automatically

def main():
    edge_giv_path = os.path.join(NETWORK_DIR, "Edge.giv")

    if GENERATE_SCENARIO_INFRA:
        print("=== Generating scenario_infra files ===")
        generate_scenario_infra_files(edge_giv_path, GENERATED_DIR)

    if GENERATE_CONFIG:
        print("\n=== Generating config.csv ===")
        generate_config(edge_giv_path, CONFIG_OUTPUT)

    print("\nDone!")

if __name__ == "__main__":
    main()

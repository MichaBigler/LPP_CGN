# -*- coding: utf-8 -*-
"""Read Data/config.csv and call load_and_build for each row."""
import os, json, pandas as pd
from load_data import load_and_build
from print import print_domain_summary, print_model_summary

def main():
    data_root = os.environ.get("DATA_ROOT", ".")
    cfg_path = os.path.join(data_root, "Data", "config.csv")
    cfg = pd.read_csv(cfg_path, sep=';')

    results = []
    for i, row in cfg.iterrows():
        domain, model = load_and_build(
            data_root=data_root,
            cfg_row=row.to_dict(),
            symmetrise_infra=False,
            zero_od_diagonal=False,
        )

        # ---- prints for verification ----
        print(f"\n\n##### RUN {i}: {row['source']}/{row['network']}@{row['scenario_line_data']} #####")
        print_domain_summary(domain, max_rows=5, max_lines=5, max_include=5)
        print_model_summary(model, domain, max_items=5, max_lines=5)

        results.append({
            "row": int(i),
            "tag": f"{row['source']}/{row['network']}@{row['scenario_line_data']}",
            "N": model.N, "E_dir": model.E_dir, "L": model.L, "S": model.S
        })

    out_path = os.path.join(data_root, "Data", "batch_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nWrote: {out_path}")

if __name__ == "__main__":
    main()

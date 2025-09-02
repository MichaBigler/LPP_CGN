import gurobipy as gp

def dump_iis(m: gp.Model, *, max_print=40):
    m.computeIIS()
    # Dateien schreiben (gültige Endungen)
    m.write("model.lp")    # gesamtes Modell
    m.write("model.ilp")   # nur IIS

    # kompakte Konsolenübersicht
    cons = m.getConstrs()
    iis_cons = [c for c in cons if c.IISConstr]
    print(f"IIS: {len(iis_cons)} linear constraints")
    # ein paar Beispiele
    for nm in [c.ConstrName for c in iis_cons[:max_print]]:
        print("  ", nm)

    # auch Variablenbounds können im IIS sein:
    lb_vars = [v.VarName for v in m.getVars() if getattr(v, "IISLB", False)]
    ub_vars = [v.VarName for v in m.getVars() if getattr(v, "IISUB", False)]
    if lb_vars or ub_vars:
        print(f"IIS-LB vars: {len(lb_vars)}, IIS-UB vars: {len(ub_vars)}")

"""
Larger-instance test for the GAP B&P.
Compare HiGHS vs Gurobi on master + pricing, validate against compact MIP.
"""
import time
from bnp import random_instance, BranchAndPrice, solve_compact


def run(label, inst, solver, max_nodes=2000):
    print(f"\n{'='*72}")
    print(f"  {label}  (solver = {solver})")
    print(f"{'='*72}")
    bp = BranchAndPrice(inst, solver=solver, max_nodes=max_nodes, verbose=True)
    t0 = time.time()
    obj, y = bp.solve()
    elapsed = time.time() - t0
    return {
        "solver": solver,
        "obj": obj,
        "time": elapsed,
        "nodes": bp.nodes_explored,
        "cg_iters": bp.cg_iters_total,
        "lp_solves": bp.lp_solves,
        "pricing_solves": bp.pricing_solves,
        "columns": len(bp.patterns),
    }


def bench(m, n, capacity_factor, seed, max_nodes=2000):
    inst = random_instance(m=m, n=n, seed=seed, capacity_factor=capacity_factor)
    print(f"\n>>> Instance: m={m}, n={n}, capacity_factor={capacity_factor}, seed={seed}")
    print(f"    capacities d: {inst.d}")
    print(f"    sum(w) = {sum(inst.w.values())}, sum(d) = {sum(inst.d.values())}")

    print("\n--- Compact MIP oracle (Gurobi) ---")
    t0 = time.time()
    compact_obj, _ = solve_compact(inst, solver="gurobi")
    print(f"Compact optimum = {compact_obj:.4f}   ({time.time()-t0:.2f}s)")

    r1 = run("B&P with HiGHS",  inst, "highs",  max_nodes=max_nodes)
    r2 = run("B&P with Gurobi", inst, "gurobi", max_nodes=max_nodes)

    print(f"\n{'='*72}")
    print(f"SUMMARY  m={m}, n={n}, cf={capacity_factor}, seed={seed}")
    print(f"{'='*72}")
    print(f"{'metric':<22}{'HiGHS':>16}{'Gurobi':>16}")
    print("-"*72)
    rows = [
        ("optimum",            f"{r1['obj']:.4f}",        f"{r2['obj']:.4f}"),
        ("wall time (s)",      f"{r1['time']:.2f}",       f"{r2['time']:.2f}"),
        ("nodes explored",     f"{r1['nodes']}",          f"{r2['nodes']}"),
        ("CG iterations",      f"{r1['cg_iters']}",       f"{r2['cg_iters']}"),
        ("LP solves (master)", f"{r1['lp_solves']}",      f"{r2['lp_solves']}"),
        ("pricing solves",     f"{r1['pricing_solves']}", f"{r2['pricing_solves']}"),
        ("columns generated",  f"{r1['columns']}",        f"{r2['columns']}"),
    ]
    for lbl, a, b in rows:
        print(f"{lbl:<22}{a:>16}{b:>16}")
    print("-"*72)
    print(f"compact MIP optimum  = {compact_obj:.4f}")
    print(f"  HiGHS  match : {abs(r1['obj']-compact_obj)<1e-4}")
    print(f"  Gurobi match : {abs(r2['obj']-compact_obj)<1e-4}")


if __name__ == "__main__":
    bench(m=8, n=40, capacity_factor=0.30, seed=11, max_nodes=2000)

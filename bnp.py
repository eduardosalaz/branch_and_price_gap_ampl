"""
Branch-and-price for the Generalized Assignment Problem (GAP)
=============================================================

Notation matches the lecture (Flamand, "Branch-and-Price"):

    N = TASKS,    indexed by i      (n tasks)
    M = MACHINES, indexed by j      (m machines)
    p[i,j]  : profit of assigning task i to machine j   (slide 14, eq (8))
    w[i,j]  : capacity of machine j consumed by task i
    d[j]    : capacity of machine j
    y[i,j]  : original binary -- 1 if task i assigned to machine j

GAP is profit-maximization throughout this codebase; we do NOT negate
to a min problem. Every sign and inequality follows the lecture as-is.

GAP (eq (8), slide 14):

    max  sum_{i in N, j in M} p[i,j] y[i,j]
    s.t. sum_{j in M} y[i,j] = 1                      for all tasks i
         sum_{i in N} w[i,j] y[i,j] <= d[j]           for all machines j
         y[i,j] in {0, 1}

Dantzig-Wolfe decomposition by machine (slides 15-16). For each machine
j, the polytope of feasible task subsets ("patterns" / "columns" h in C_j)
gives:

    Master (eq (9), (10)):
        max  sum_{j,h} pi_{j,h} x[j,h]
        s.t. sum_{j,h} a_{j,i,h} x[j,h] = 1   for each task i     (dual pi[i])
             sum_h     x[j,h]            = 1   for each machine j  (dual mu[j])
             x[j,h] >= 0                          (LP relaxation)

    Pricing for machine j (eq (11)):
        SP(j, pi, mu):  max  sum_i (p[i,j] - pi[i]) z[i]  -  mu[j]
                        s.t. sum_i w[i,j] z[i] <= d[j]
                             z[i] in {0, 1}

A column with SP objective > 0 is improving and is added to the master.

Branching (slides 24-25, 27): on the original variable
    y[i,j] = sum_{h: a_{j,i,h} = 1} x[j,h].
The most-fractional y[i,j] is selected; the up child forces y[i,j] = 1,
the down child forces y[i,j] = 0. These branching constraints are
"compatible with pricing" -- they reduce to variable fixings:
    master side : incompatible existing patterns get x[h] fixed to 0,
    pricing side: z[i] is fixed via fix_in / fix_out per machine.
"""

from amplpy import AMPL
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional
import math
import os
import random
import sys
import time


@contextmanager
def _silenced():
    """Suppress C-level stdout/stderr (solver banners) and Python-level prints
    coming from amplpy subprocesses."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved_out, saved_err = os.dup(1), os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        os.close(devnull)
        os.dup2(saved_out, 1); os.close(saved_out)
        os.dup2(saved_err, 2); os.close(saved_err)


# =====================================================================
# Data structures (lecture notation)
# =====================================================================

@dataclass(frozen=True)
class Pattern:
    """One column h of machine j in the master problem.

    tasks  : the set of task indices i with a_{j,i,h} = 1.
    profit : the column reward pi_{j,h} = sum_{i in tasks} p[i, j].
    """
    pid: int
    machine: int        # j
    tasks: frozenset    # {i : a_{j,i,h} = 1}
    profit: float       # pi_{j,h}


@dataclass
class GAPInstance:
    """GAP data in lecture notation (slide 14)."""
    tasks: list             # N
    machines: list          # M
    p: dict                 # (i, j) -> p[i, j]
    w: dict                 # (i, j) -> w[i, j]
    d: dict                 # j      -> d[j]


@dataclass(frozen=True)
class Branch:
    """A single branching decision  y[i, j] = value  (slide 25)."""
    task: int           # i
    machine: int        # j
    value: int          # 1 (force) or 0 (forbid)


@dataclass
class Node:
    branches: tuple = ()
    ub: float = math.inf    # upper bound from parent's LP (max problem)
    depth: int = 0

    def forced(self):
        return frozenset((b.task, b.machine) for b in self.branches if b.value == 1)

    def forbidden(self):
        return frozenset((b.task, b.machine) for b in self.branches if b.value == 0)


# =====================================================================
# Branch-and-price algorithm
# =====================================================================

class BranchAndPrice:

    EPS = 1e-6

    def __init__(self, instance: GAPInstance,
                 master_mod="master.mod",
                 pricing_mod="pricing.mod",
                 solver="highs",
                 verbose=True,
                 max_nodes=10_000):
        self.inst = instance
        self.solver = solver
        self.verbose = verbose
        self.max_nodes = max_nodes
        self.master_mod = master_mod
        self.pricing_mod = pricing_mod

        # Global pool of all generated columns (kept across nodes).
        self.patterns: dict = {}
        self.next_pid = 0

        # Best-known integer feasible solution (max problem -> -inf start).
        self.incumbent_profit = -math.inf
        self.incumbent_y: Optional[dict] = None

        self.bigM = self._compute_bigM()
        self._build_master()
        self._build_pricers()

        # Empty pattern per machine so the convexity equality is satisfiable
        # in the very first LP solve. Profit zero, no tasks. Cover constraints
        # are then closed by Slack at cost -BigM per task.
        for j in self.inst.machines:
            self._add_and_register(machine=j, tasks=frozenset(), profit=0.0)

        self.nodes_explored = 0
        self.cg_iters_total = 0
        self.lp_solves = 0
        self.pricing_solves = 0

    # -----------------------------------------------------------------
    # Setup
    # -----------------------------------------------------------------

    def _compute_bigM(self):
        """A loose upper bound on any feasible total profit, scaled so the
        -BigM * Slack penalty in the master always dominates any column
        profit gain. Ensures the LP prefers covering tasks over leaving
        slack > 0 whenever a feasible covering pattern exists."""
        worst = sum(max(self.inst.p[i, j] for j in self.inst.machines)
                    for i in self.inst.tasks)
        return 100.0 * (worst + 1.0)

    def _build_master(self):
        m = AMPL()
        m.read(self.master_mod)
        m.set["TASKS"]    = self.inst.tasks
        m.set["MACHINES"] = self.inst.machines
        m.param["BigM"] = self.bigM
        m.option["solver"] = self.solver
        m.option["solver_msg"] = 0
        m.option[f"{self.solver}_options"] = "outlev=0"
        self.master = m

    def _build_pricers(self):
        """One persistent AMPL object per machine's pricing knapsack."""
        self.pricers = {}
        for j in self.inst.machines:
            pr = AMPL()
            pr.read(self.pricing_mod)
            pr.set["TASKS"] = self.inst.tasks
            pr.param["w"] = {i: float(self.inst.w[i, j]) for i in self.inst.tasks}
            pr.param["d"] = float(self.inst.d[j])
            pr.option["solver"] = self.solver
            pr.option["solver_msg"] = 0
            pr.option[f"{self.solver}_options"] = "outlev=0"
            self.pricers[j] = pr

    # -----------------------------------------------------------------
    # Pattern bookkeeping
    # -----------------------------------------------------------------

    def _add_and_register(self, machine, tasks, profit):
        pid = self.next_pid
        self.next_pid += 1
        pat = Pattern(pid=pid, machine=machine, tasks=tasks, profit=profit)
        self.patterns[pid] = pat
        m = self.master
        m.eval(f"let PATTERNS := PATTERNS union {{{pid}}};")
        m.eval(f"let pat_machine[{pid}] := {machine};")
        m.eval(f"let pat_profit[{pid}] := {profit};")
        for i in tasks:
            m.eval(f"let pat_covers[{pid},{i}] := 1;")
        return pid

    def _has_duplicate(self, machine, tasks):
        for pat in self.patterns.values():
            if pat.machine == machine and pat.tasks == tasks:
                return True
        return False

    # -----------------------------------------------------------------
    # Branching context translated to variable fixings
    # -----------------------------------------------------------------

    @staticmethod
    def _is_compatible(pat: Pattern, forced, forbidden):
        """True if this existing column is allowed under the node's
        branching context. forced / forbidden are sets of (task, machine)."""
        for (i, j) in forbidden:
            # y[i, j] = 0 forbids any column on machine j that covers task i.
            if pat.machine == j and i in pat.tasks:
                return False
        for (i, j) in forced:
            # y[i, j] = 1 requires task i be in the chosen column of machine j,
            # and forbids it from any column of any other machine.
            if pat.machine == j and i not in pat.tasks:
                return False
            if pat.machine != j and i in pat.tasks:
                return False
        return True

    def _apply_node_context(self, node: Node):
        """Walk the global pattern pool and pin/unpin master columns to
        match the node's branching path. Uses AMPL's runtime fix/unfix."""
        forced = node.forced()
        forbidden = node.forbidden()
        for pid, pat in self.patterns.items():
            if self._is_compatible(pat, forced, forbidden):
                self.master.eval(f"unfix X[{pid}];")
            else:
                self.master.eval(f"fix X[{pid}] := 0;")

    def _machine_branching_constraints(self, machine, forced, forbidden):
        """Translate the node's branching path into (fix_in, fix_out)
        sets of tasks for THIS machine's pricing problem."""
        fix_in, fix_out = set(), set()
        for (i, j) in forced:
            if j == machine:
                fix_in.add(i)
            else:
                # Task i is committed to machine j != this one; this
                # machine cannot take it.
                fix_out.add(i)
        for (i, j) in forbidden:
            if j == machine:
                fix_out.add(i)
            # forbidden on a different machine does not constrain this one.
        return fix_in, fix_out

    # -----------------------------------------------------------------
    # Column generation
    # -----------------------------------------------------------------

    def _solve_master(self):
        self.lp_solves += 1
        self.master.solve()
        return self.master.solve_result == "solved"

    def _get_duals(self):
        """Return (pi, mu): pi[i] = CoverTask[i].dual, mu[j] = Convexity[j].dual."""
        pi_df = self.master.get_data(
            "{i in TASKS} CoverTask[i].dual"
        ).to_pandas()
        mu_df = self.master.get_data(
            "{j in MACHINES} Convexity[j].dual"
        ).to_pandas()
        pi = {idx: float(pi_df.iloc[k, 0]) for k, idx in enumerate(pi_df.index)}
        mu = {idx: float(mu_df.iloc[k, 0]) for k, idx in enumerate(mu_df.index)}
        return pi, mu

    def _solve_pricing(self, machine, pi, mu_j, fix_in, fix_out):
        """Solve SP(machine, pi, mu) (lecture eq (11)). Return
        (tasks, profit, reduced_profit) for an improving column, else None.

        The AMPL pricing maximizes sum_i (p[i,j] - pi[i]) Z[i] (knap_obj);
        the lecture's reduced profit is knap_obj - mu[j]. A column is
        improving when reduced_profit > 0."""
        pr = self.pricers[machine]
        red = {i: float(self.inst.p[i, machine] - pi[i]) for i in self.inst.tasks}
        pr.param["red_profit"] = red
        pr.param["fix_in"]  = {i: (1 if i in fix_in  else 0) for i in self.inst.tasks}
        pr.param["fix_out"] = {i: (1 if i in fix_out else 0) for i in self.inst.tasks}
        self.pricing_solves += 1
        pr.solve()
        if pr.solve_result != "solved":
            return None
        knap_obj = pr.get_objective("ReducedProfit").value()
        reduced = knap_obj - mu_j
        if reduced > self.EPS:
            Z_df = pr.get_data("{i in TASKS} Z[i]").to_pandas()
            tasks_in = frozenset(
                idx for k, idx in enumerate(Z_df.index)
                if Z_df.iloc[k, 0] > 0.5
            )
            profit = sum(self.inst.p[i, machine] for i in tasks_in)
            return tasks_in, profit, reduced
        return None

    def _column_generation(self, node: Node):
        """Iterate master <-> pricing until no machine has an improving
        column. Returns the LP master objective value, or None if the
        master LP is infeasible at this node."""
        self._apply_node_context(node)
        forced, forbidden = node.forced(), node.forbidden()

        while True:
            self.cg_iters_total += 1
            if not self._solve_master():
                return None
            pi, mu = self._get_duals()
            added = 0
            for j in self.inst.machines:
                fix_in, fix_out = self._machine_branching_constraints(j, forced, forbidden)
                res = self._solve_pricing(j, pi, mu[j], fix_in, fix_out)
                if res is None:
                    continue
                tasks_in, profit, _ = res
                if self._has_duplicate(j, tasks_in):
                    continue
                self._add_and_register(j, tasks_in, profit)
                # Newly priced columns respect the current branching context
                # by construction (fix_in / fix_out enforced it), so they
                # are created unfixed by default.
                added += 1
            if added == 0:
                break
        return self.master.get_objective("MasterProfit").value()

    # -----------------------------------------------------------------
    # Branching on the original variable y[i, j]
    # -----------------------------------------------------------------

    def _compute_y(self):
        """Recover the original GAP variable from the LP master:
            y[i, j] = sum_{h on machine j with task i in column h} x[j, h]."""
        X_df = self.master.get_data("{h in PATTERNS} X[h]").to_pandas()
        y = {}
        for k, pid in enumerate(X_df.index):
            val = float(X_df.iloc[k, 0])
            if val < self.EPS:
                continue
            pat = self.patterns[pid]
            for i in pat.tasks:
                y[(i, pat.machine)] = y.get((i, pat.machine), 0.0) + val
        return y

    def _branch(self, y):
        """Most-fractional rule on y[i, j] (lecture slide 25):
            argmax_{(i, j)} min(y[i, j], 1 - y[i, j]).
        Returns (i, j, distance), or None if y is integer."""
        best, best_dist = None, -1.0
        for (i, j), v in y.items():
            d = min(v, 1.0 - v)
            if d > best_dist + self.EPS:
                best_dist = d
                best = (i, j)
        if best is None or best_dist < self.EPS:
            return None
        return (best[0], best[1], best_dist)

    def _slack_active(self):
        """Any leftover Slack > 0 in the optimal LP means the cover
        constraints could not be closed by feasible columns at this
        node -- branching has rendered the subproblem infeasible."""
        slk_df = self.master.get_data("{i in TASKS} Slack[i]").to_pandas()
        for k in range(len(slk_df)):
            if float(slk_df.iloc[k, 0]) > self.EPS:
                return True
        return False

    def _extract_integer_assignment(self, y):
        return {(i, j): 1 for (i, j), v in y.items() if v > 0.5}

    # -----------------------------------------------------------------
    # Main DFS loop
    # -----------------------------------------------------------------

    def solve(self):
        t0 = time.time()
        root = Node(branches=(), ub=math.inf, depth=0)
        stack = [root]

        while stack:
            if self.nodes_explored >= self.max_nodes:
                self._log(f"hit max_nodes={self.max_nodes}; stopping.")
                break
            node = stack.pop()
            self.nodes_explored += 1

            # Bound prune (max problem): parent's LP value is an upper
            # bound on this subtree's IP optimum; if it does not beat
            # the incumbent, no need to even run column generation.
            if node.ub <= self.incumbent_profit + self.EPS:
                continue

            lp_obj = self._column_generation(node)
            if lp_obj is None:
                self._log(f"  node {self.nodes_explored} (d={node.depth}): master infeasible")
                continue
            if self._slack_active():
                self._log(f"  node {self.nodes_explored} (d={node.depth}): infeasible (slack active)")
                continue
            if lp_obj <= self.incumbent_profit + self.EPS:
                self._log(f"  node {self.nodes_explored} (d={node.depth}): "
                          f"pruned, LP={lp_obj:.4f} <= LB={self.incumbent_profit:.4f}")
                continue

            y = self._compute_y()
            br = self._branch(y)
            if br is None:
                # LP solution is already integer in y => GAP-feasible.
                if lp_obj > self.incumbent_profit + self.EPS:
                    self.incumbent_profit = lp_obj
                    self.incumbent_y = self._extract_integer_assignment(y)
                    self._log(f"  node {self.nodes_explored} (d={node.depth}): "
                              f"NEW INCUMBENT {lp_obj:.4f}")
                continue

            i, j, dist = br
            self._log(f"  node {self.nodes_explored} (d={node.depth}): "
                      f"LP={lp_obj:.4f}, branch y[{i},{j}] (dist={dist:.3f})")
            up   = Node(branches=node.branches + (Branch(i, j, 1),),
                        ub=lp_obj, depth=node.depth + 1)
            down = Node(branches=node.branches + (Branch(i, j, 0),),
                        ub=lp_obj, depth=node.depth + 1)
            # DFS: explore down child first (push up first so down pops first).
            stack.append(up)
            stack.append(down)

        elapsed = time.time() - t0
        self._log("")
        self._log(f"== B&P done ==")
        self._log(f"  nodes explored : {self.nodes_explored}")
        self._log(f"  CG iterations  : {self.cg_iters_total}")
        self._log(f"  LP solves      : {self.lp_solves}")
        self._log(f"  pricing solves : {self.pricing_solves}")
        self._log(f"  columns total  : {len(self.patterns)}")
        self._log(f"  best profit    : {self.incumbent_profit:.4f}")
        self._log(f"  wall time      : {elapsed:.2f}s")
        return self.incumbent_profit, self.incumbent_y

    def _log(self, msg):
        if self.verbose:
            print(msg)


# =====================================================================
# Validation: compact GAP MIP (lecture eq (8))
# =====================================================================

def solve_compact(instance: GAPInstance, model_path="compact.mod", solver="highs"):
    ampl = AMPL()
    ampl.read(model_path)
    ampl.set["TASKS"]    = instance.tasks
    ampl.set["MACHINES"] = instance.machines
    ampl.param["p"] = instance.p
    ampl.param["w"] = instance.w
    ampl.param["d"] = instance.d
    ampl.option["solver"] = solver
    ampl.option["solver_msg"] = 0
    ampl.option[f"{solver}_options"] = "outlev=0"
    ampl.solve()
    assert ampl.solve_result == "solved", ampl.solve_result
    obj = ampl.get_objective("Total").value()
    Y_df = ampl.get_data("{i in TASKS, j in MACHINES} Y[i,j]").to_pandas()
    assignment = {}
    for k in range(len(Y_df)):
        idx = Y_df.index[k]
        if isinstance(idx, tuple):
            i, j = idx
        else:
            i, j = idx
        if float(Y_df.iloc[k, 0]) > 0.5:
            assignment[(i, j)] = 1
    return obj, assignment


# =====================================================================
# Demo instances
# =====================================================================

def random_instance(m=3, n=10, seed=42, capacity_factor=0.6):
    """Random GAP instance with m machines and n tasks.

    capacity_factor : d[j] = capacity_factor * sum_i w[i, j].
                      Lower => tighter capacity, more interesting tree.
    """
    rng = random.Random(seed)
    machines = list(range(1, m + 1))
    tasks    = list(range(1, n + 1))
    p = {(i, j): rng.randint(10, 50) for i in tasks for j in machines}
    w = {(i, j): rng.randint(5, 25)  for i in tasks for j in machines}
    d = {j: int(capacity_factor * sum(w[i, j] for i in tasks)) for j in machines}
    return GAPInstance(tasks=tasks, machines=machines, p=p, w=w, d=d)


def lecture_slide_18_instance():
    """Worked example from lecture slides 18-23 (2 machines, 3 tasks).

    Expected optimum on slide 23: z* = 17, attained by
        machine 1 takes task {2}      -> profit p[2,1] = 3
        machine 2 takes tasks {1, 3}  -> profit p[1,2] + p[3,2] = 4 + 10 = 14
    """
    return GAPInstance(
        tasks=[1, 2, 3],
        machines=[1, 2],
        p={
            (1, 1): 1, (1, 2): 4,
            (2, 1): 3, (2, 2): 5,
            (3, 1): 8, (3, 2): 10,
        },
        w={
            (1, 1): 4, (1, 2): 3,
            (2, 1): 5, (2, 2): 5,
            (3, 1): 3, (3, 2): 2,
        },
        d={1: 8, 2: 5},
    )


def symmetric_instance():
    """Designed to have a fractional LP at the root so branching fires.
    2 identical machines, 3 identical tasks, each machine fits 2 tasks.
    LP root has y[i, j] = 0.5 for all i, j."""
    return GAPInstance(
        tasks=[1, 2, 3],
        machines=[1, 2],
        p={(i, j): 10 for i in [1, 2, 3] for j in [1, 2]},
        w={(i, j): 2  for i in [1, 2, 3] for j in [1, 2]},
        d={1: 4, 2: 4},
    )


# =====================================================================
# Entry point
# =====================================================================

def _print_solution(inst, y):
    if not y:
        return
    by_machine = {}
    for (i, j), _ in y.items():
        by_machine.setdefault(j, []).append(i)
    for j in sorted(by_machine):
        tasks_j = sorted(by_machine[j])
        load = sum(inst.w[i, j] for i in tasks_j)
        print(f"  machine {j}: tasks {tasks_j}  load={load}/{inst.d[j]}")


def _run_demo(inst, label, expected_z=None):
    print(f"========== {label} ==========")
    print(f"Instance: |N| = {len(inst.tasks)} tasks, |M| = {len(inst.machines)} machines")
    print(f"Capacities d: {inst.d}")
    print()

    print("--- Branch-and-Price ---")
    bnp = BranchAndPrice(inst, verbose=True)
    bnp_obj, bnp_y = bnp.solve()

    print()
    print("--- Compact MIP (validation oracle) ---")
    compact_obj, _ = solve_compact(inst)
    print(f"Compact optimum: {compact_obj:.4f}")
    print()

    match = abs(bnp_obj - compact_obj) < 1e-4
    print(f"B&P vs compact match: {match}")
    if expected_z is not None:
        slide_match = abs(bnp_obj - expected_z) < 1e-4
        print(f"Lecture slide says z* = {expected_z}; B&P z = {bnp_obj:.4f}; match = {slide_match}")
    if bnp_y:
        print("B&P assignment by machine:")
        _print_solution(inst, bnp_y)
    print()


if __name__ == "__main__":
    # Demo 1: lecture's slide 18-23 worked example. Optimal z* = 17.
    _run_demo(lecture_slide_18_instance(),
              "Demo 1: lecture slide 18 worked example",
              expected_z=17)

    # Demo 2: symmetric instance designed so the LP root is fractional
    # and branching actually fires (2 identical machines, 3 identical
    # tasks, each machine fits 2 tasks; LP root has y[i,j] = 0.5).
    _run_demo(symmetric_instance(),
              "Demo 2: symmetric (fractional LP root, branching)")

    # Demo 3: a larger random instance. Column generation typically
    # closes the gap at the root for asymmetric GAP instances -- a
    # known strength of Dantzig-Wolfe over the compact LP relaxation.
    rnd = random_instance(m=4, n=15, seed=11, capacity_factor=0.40)
    _run_demo(rnd, "Demo 3: random asymmetric (root LP usually integer)")
